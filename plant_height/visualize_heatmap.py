"""Visualize YOLO26 head feature maps as heatmaps before detection.

Usage:
    python visualize_heatmap.py --input image.jpg --save heatmap_output.jpg
    python visualize_heatmap.py --input image.jpg --save heatmap_output.jpg --model path/to/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


def register_hook(model: nn.Module, features: dict) -> list:
    """Register forward hooks on head layers (layers 16, 19, 22 in YOLO26-pose).

    These layers produce the P3/P4/P5 feature maps that are fed into the Pose26 head.
    """
    hooks = []
    head_layer_indices = [16, 19, 22]  # P3-small, P4-medium, P5-large

    for idx in head_layer_indices:
        layer = model.model.model[idx]

        def make_hook(i):
            def hook(module, input, output):
                features[i] = output.detach().cpu()
            return hook

        hooks.append(layer.register_forward_hook(make_hook(idx)))
    return hooks


def remove_hooks(hooks: list):
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


def feature_to_heatmap(feat: torch.Tensor, target_size: tuple) -> np.ndarray:
    """Convert a feature map tensor (C, H, W) to a colored heatmap image.

    Averages across channels, normalizes, and applies a colormap.
    """
    # Average across channels
    heatmap = feat.mean(dim=0).numpy()  # (H, W)
    # Normalize to 0-255
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)
    # Resize to target size
    heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]))
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on the original image."""
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO26 head feature maps as heatmaps")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--save", type=str, required=True, help="Path to save heatmap output")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (default: auto-detect)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha (default: 0.5)")
    parser.add_argument("--no-overlay", action="store_true", help="Save heatmap only without original image overlay")
    args = parser.parse_args()

    # Load model
    if args.model:
        model = YOLO(args.model)
    else:
        # Auto-detect best weights from training runs
        run_dir = Path(__file__).resolve().parent / "../runs"
        best_pt = None
        for d in sorted(run_dir.glob("plant_height_yolo26*"), reverse=True):
            candidate = d / "weights" / "best.pt"
            if candidate.exists():
                best_pt = str(candidate)
                break
        if best_pt is None:
            best_pt = str(Path(__file__).resolve().parent / "../data/yolo26m-pose.pt")
        print(f"Using model: {best_pt}")
        model = YOLO(best_pt)

    # Read original image
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    h, w = image_bgr.shape[:2]

    # Register hooks to capture feature maps before head
    features = {}
    hooks = register_hook(model, features)

    # Forward pass to trigger hooks
    with torch.no_grad():
        model.predict(args.input, imgsz=640, verbose=False)

    remove_hooks(hooks)

    if not features:
        print("ERROR: No feature maps captured. Check model architecture.")
        return

    # Generate and save heatmaps separately: {save}_p3.jpg, {save}_p4.jpg, {save}_p5.jpg
    layer_info = {16: ("p3", "P3/8-small"), 19: ("p4", "P4/16-medium"), 22: ("p5", "P5/32-large")}
    save_path = Path(args.save)
    if save_path.suffix:
        save_dir, stem, suffix = save_path.parent, save_path.stem, save_path.suffix
    else:
        save_dir = save_path
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
        suffix = ".jpg"

    for idx in sorted(features.keys()):
        feat = features[idx][0]  # (C, H, W)
        heatmap = feature_to_heatmap(feat, (h, w))
        tag, name = layer_info.get(idx, (f"layer{idx}", f"Layer-{idx}"))
        if not args.no_overlay:
            output = overlay_heatmap(image_bgr, heatmap, alpha=args.alpha)
        else:
            output = heatmap
        cv2.putText(output, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out_file = save_dir / f"{stem}_{tag}{suffix}"
        cv2.imwrite(str(out_file), output)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
