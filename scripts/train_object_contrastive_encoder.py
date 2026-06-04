from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir


class CropDataset:
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        from PIL import Image

        row = self.frame.iloc[index]
        image = Image.open(row["crop_path"]).convert("RGB")
        return self.transform(image), int(row["class_index"]), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple supervised-contrastive crop encoder.")
    parser.add_argument("--crops-metadata", required=True)
    parser.add_argument("--output-dir", default="outputs/contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from tqdm import tqdm

    output_dir = ensure_dir(args.output_dir)
    metadata = pd.read_csv(args.crops_metadata)
    metadata = metadata[metadata["crop_path"].map(lambda value: Path(value).exists())].reset_index(drop=True)
    if metadata.empty:
        raise RuntimeError("No crop files found. Run extract_object_crops.py first.")

    class_encoder = LabelEncoder()
    metadata["class_index"] = class_encoder.fit_transform(metadata["class_name"].astype(str))

    transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ]
    )

    class Encoder(nn.Module):
        def __init__(self, embedding_dim: int):
            super().__init__()
            backbone = models.resnet18(weights=None)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.projection = nn.Linear(in_features, embedding_dim)

        def forward(self, x):
            return F.normalize(self.projection(self.backbone(x)), dim=1)

    def supcon_loss(features, labels, temperature: float = 0.1):
        features = F.normalize(features, dim=1)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.div(torch.matmul(features, features.T), temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_counts = mask.sum(dim=1)
        valid = positive_counts > 0
        if not valid.any():
            return features.sum() * 0.0
        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[valid] / positive_counts[valid]
        return -mean_log_prob_pos.mean()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device) != "cpu" else "cpu")
    dataset = CropDataset(metadata, transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    model = Encoder(args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = supcon_loss(embeddings, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

    torch.save({"model": model.state_dict(), "classes": class_encoder.classes_.tolist(), "embedding_dim": args.embedding_dim}, output_dir / "encoder.pt")

    eval_dataset = CropDataset(metadata, eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    embeddings = np.zeros((len(eval_dataset), args.embedding_dim), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for images, _, indices in tqdm(eval_loader, desc="Embedding crops"):
            images = images.to(device)
            batch_embeddings = model(images).cpu().numpy()
            embeddings[indices.numpy()] = batch_embeddings

    np.save(output_dir / "embeddings.npy", embeddings)
    metadata.drop(columns=["class_index"]).to_csv(output_dir / "embeddings_metadata.csv", index=False)

    if len(metadata) >= 2:
        perplexity = max(1, min(30, len(metadata) - 1))
        coords = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto", random_state=42).fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(6, 5))
        for class_name in sorted(metadata["class_name"].astype(str).unique()):
            mask = metadata["class_name"].astype(str) == class_name
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, label=class_name, alpha=0.75)
        ax.set_title("Object Crop Embeddings (t-SNE)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "tsne_by_class.png", dpi=180)
        plt.close(fig)

        try:
            import umap  # type: ignore

            reducer = umap.UMAP(random_state=42)
            umap_coords = reducer.fit_transform(embeddings)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(umap_coords[:, 0], umap_coords[:, 1], s=8, c=metadata["class_index"], cmap="tab20", alpha=0.75)
            ax.set_title("Object Crop Embeddings (UMAP)")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(output_dir / "umap_by_class.png", dpi=180)
            plt.close(fig)
        except Exception as exc:
            print(f"UMAP skipped: {exc}")

    print(f"Contrastive outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
