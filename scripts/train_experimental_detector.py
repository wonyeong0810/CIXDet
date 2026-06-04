from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental detector-level contrastive learning skeleton for CI-XDet."
    )
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--lambda-contrast", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    parse_args()
    raise NotImplementedError(
        "Detector-level contrastive learning is intentionally left as a research TODO. "
        "A future version would add L_total = L_det + lambda_contrast * L_object_contrast "
        "around object/ROI features without modifying the stable YOLO baseline scripts."
    )


if __name__ == "__main__":
    main()

