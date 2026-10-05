import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from infer_sam3 import compute_outline, to_normalized_box
from model.sam3_wrapper import build_sam3_segmentation_model


def parse_args():
    parser = argparse.ArgumentParser("SAM3 inference for a directory of images")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--seg-checkpoint", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--box", required=True, nargs=4, type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Shared prompt box in normalized [0,1] coordinates.",
    )
    parser.add_argument("--box-pixels", action="store_true")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if not args.images_dir.is_dir():
        parser.error("--images-dir must be an existing directory")
    if not args.weights.is_file():
        parser.error("--weights must be an existing file")
    if args.seg_checkpoint is not None and not args.seg_checkpoint.is_file():
        parser.error("--seg-checkpoint must be an existing file")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if not all(np.isfinite(value) for value in args.box):
        parser.error("--box coordinates must be finite")
    x1, y1, x2, y2 = args.box
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        parser.error("--box requires 0 <= X1 < X2 and 0 <= Y1 < Y2")
    if not args.box_pixels and max(args.box) > 1:
        parser.error("Normalized --box coordinates must be <= 1")
    return args


def main():
    args = parse_args()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == images_dir:
        raise ValueError("--output-dir must differ from --images-dir")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    candidates = images_dir.rglob("*") if args.recursive else images_dir.iterdir()
    image_paths = sorted(
        path for path in candidates
        if path.is_file() and path.suffix.lower() in extensions
        and output_dir not in path.parents
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; using CPU")
    model = build_sam3_segmentation_model(
        checkpoint=str(args.seg_checkpoint) if args.seg_checkpoint is not None else None,
        weights_path=str(args.weights),
        num_classes=1,
    ).to(device)
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        box = to_normalized_box(args.box, width, height, args.box_pixels)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"Prompt box is outside image bounds: {image_path}")
        box_tensor = torch.tensor([box], dtype=torch.float32, device=device)
        with torch.no_grad():
            mask_logits, scores = model(preprocess(image).unsqueeze(0).to(device), box_tensor)
            mask_prob = torch.sigmoid(mask_logits[0, 0]).cpu()
            mask_np = ((mask_prob > args.threshold).to(torch.uint8) * 255).numpy()

        relative_path = image_path.relative_to(images_dir)
        result_dir = output_dir / relative_path.parent
        result_dir.mkdir(parents=True, exist_ok=True)
        mask_path = result_dir / f"{relative_path.name}_mask.png"
        overlay_path = result_dir / f"{relative_path.name}_overlay.png"
        Image.fromarray(mask_np).save(mask_path)

        overlay = np.array(image.convert("RGBA"))
        foreground = mask_np > 0
        overlay[foreground, 0] = 255
        overlay[foreground, 1:3] = overlay[foreground, 1:3] // 2
        overlay[compute_outline(mask_np)] = [0, 255, 255, 255]
        Image.fromarray(overlay).save(overlay_path)
        print(f"[{index}/{len(image_paths)}] {relative_path}: score={scores.item():.4f}")

    print(f"Saved {len(image_paths)} masks and overlays to {output_dir}")


if __name__ == "__main__":
    main()