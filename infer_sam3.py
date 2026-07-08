import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model.sam3_wrapper import build_sam3_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser("Inference script for fine-tuned SAM3 wrapper")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to fine-tuned checkpoint (e.g. output_dir/checkpoint_best.pth)",
    )
    parser.add_argument(
        "--seg-checkpoint",
        default=None,
        type=str,
        help="Optional base SAM3 checkpoint path (leave empty to use default)",
    )
    parser.add_argument(
        "--box",
        required=True,
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Prompt box. Normalized [0,1] by default, or pixels if --box-pixels is set.",
    )
    parser.add_argument(
        "--box-pixels",
        action="store_true",
        help="Interpret --box values as pixel coordinates instead of normalized.",
    )
    parser.add_argument("--threshold", default=0.5, type=float, help="Mask threshold")
    parser.add_argument(
        "--output-mask",
        default="pred_mask.png",
        type=str,
        help="Output path for binary mask image",
    )
    parser.add_argument(
        "--output-overlay",
        default="pred_overlay.png",
        type=str,
        help="Output path for overlay image",
    )
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    return parser.parse_args()


def to_normalized_box(box, w, h, box_pixels=False):
    x1, y1, x2, y2 = box
    if box_pixels:
        x1 = x1 / w
        x2 = x2 / w
        y1 = y1 / h
        y2 = y2 / h
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return [x1, y1, x2, y2]


def compute_outline(mask_np):
    """Return a 1-pixel outline from a binary mask (uint8 0/255)."""
    m = mask_np > 0
    up = np.zeros_like(m)
    down = np.zeros_like(m)
    left = np.zeros_like(m)
    right = np.zeros_like(m)

    up[1:, :] = m[:-1, :]
    down[:-1, :] = m[1:, :]
    left[:, 1:] = m[:, :-1]
    right[:, :-1] = m[:, 1:]

    interior = m & up & down & left & right
    outline = m & ~interior
    return outline


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    x = preprocess(image).unsqueeze(0).to(device)

    box = to_normalized_box(args.box, width, height, box_pixels=args.box_pixels)
    box_tensor = torch.tensor([box], dtype=torch.float32, device=device)

    model = build_sam3_segmentation_model(
        checkpoint=args.seg_checkpoint,
        weights_path=str(weights_path),
        num_classes=1,
    ).to(device)
    model.eval()

    with torch.no_grad():
        mask_logits, scores = model(x, box_tensor)
        mask_prob = torch.sigmoid(mask_logits[0, 0]).cpu()
        mask_bin = (mask_prob > args.threshold).to(torch.uint8) * 255

    out_mask = Image.fromarray(mask_bin.numpy(), mode="L")
    out_mask.save(args.output_mask)

    # Simple red overlay
    overlay = image.copy().convert("RGBA")
    overlay_data = overlay.load()
    mask_np = mask_bin.numpy()
    outline_np = compute_outline(mask_np)
    for yy in range(mask_np.shape[0]):
        for xx in range(mask_np.shape[1]):
            if mask_np[yy, xx] > 0:
                r, g, b, _ = overlay_data[xx, yy]
                overlay_data[xx, yy] = (255, int(g * 0.5), int(b * 0.5), 255)
            if outline_np[yy, xx]:
                # Cyan outline for better contrast over red fill.
                overlay_data[xx, yy] = (0, 255, 255, 255)
    overlay.save(args.output_overlay)

    print(f"score={scores.item():.4f}")
    print(f"mask saved to {args.output_mask}")
    print(f"overlay saved to {args.output_overlay}")


if __name__ == "__main__":
    main()
