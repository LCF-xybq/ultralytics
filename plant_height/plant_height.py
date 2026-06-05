"""
plant_height.py - Plant height inference and post-processing pipeline

Workflow:
  1. Load YOLO model, run inference on images, save LabelMe JSON to --save_json
  2. Optional: point correction (ruler extraction -> gradient -> nearest-point fix), update --save_json
  3. Compute plant heights
  4. Visualize (bbox + keypoint + height) and save to --save_img

Usage:
  python plant_height.py --weight best.pt --data ./images --save_json ./results
  python plant_height.py --weight best.pt --data ./images --save_json ./results --point_correct
"""

import argparse
import json
import os
import os.path as osp
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
from ultralytics import YOLO

from utils import (
    batch_restore,
    batch_transformer,
    compute_single_image,
    extract_ruler,
    _gradient_one,
    _nearest_one,
)

KPT_NAMES = ["up_left", "up_right", "down_left", "tip"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Colors (BGR)
BBOX_COLOR = (87, 62, 226)
POINT_COLOR = (160, 123, 36)
HEIGHT_BG_COLOR = (198, 232, 255)
HEIGHT_TEXT_COLOR = (85, 85, 85)


def list_images(img_dir):
    return sorted(n for n in os.listdir(img_dir) if osp.splitext(n)[1].lower() in IMG_EXTS)


# ---------------------------------------------------------------------------
# Step 1: Inference
# ---------------------------------------------------------------------------
def _to_labelme(img_name, h, w, bbox_xyxy, kpts_xy):
    """Convert one detection to LabelMe JSON dict."""
    x1, y1, x2, y2 = map(float, bbox_xyxy.tolist())
    shapes = [
        {
            "label": "ruler",
            "points": [[x1, y1], [x2, y2]],
            "group_id": None, "description": "",
            "shape_type": "rectangle", "flags": {}, "mask": None,
        }
    ]
    for idx, name in enumerate(KPT_NAMES):
        x, y = kpts_xy[idx].tolist()
        shapes.append({
            "label": name,
            "points": [[float(x), float(y)]],
            "group_id": None, "description": "",
            "shape_type": "point", "flags": {}, "mask": None,
        })
    return {
        "version": "5.9.1", "flags": {}, "shapes": shapes,
        "imagePath": img_name, "imageHeight": int(h), "imageWidth": int(w),
    }


def _empty_labelme(img_name, h, w):
    return {
        "version": "5.9.1", "flags": {}, "shapes": [],
        "imagePath": img_name, "imageHeight": int(h), "imageWidth": int(w),
    }


def inference(weight, data_dir, json_dir, device, imgsz, max_det):
    model = YOLO(weight)
    img_names = list_images(data_dir)
    print(f"[Inference] Found {len(img_names)} images")

    os.makedirs(json_dir, exist_ok=True)

    ok, fail = 0, 0
    for img_name in img_names:
        img_fp = osp.join(data_dir, img_name)
        img = cv2.imread(img_fp)
        if img is None:
            fail += 1
            continue
        h, w = img.shape[:2]

        results = model.predict(
            source=img_fp, imgsz=imgsz, max_det=max_det,
            conf=0.25, verbose=False, device=device,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0 and r.keypoints is not None:
            confs = r.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            out = _to_labelme(
                img_name, h, w,
                r.boxes.xyxy.cpu().numpy()[best],
                r.keypoints.xy.cpu().numpy()[best],
            )
        else:
            out = _empty_labelme(img_name, h, w)

        stem = osp.splitext(img_name)[0]
        with open(osp.join(json_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        ok += 1

    print(f"[Inference] Done. ok={ok}, failed={fail}")


# ---------------------------------------------------------------------------
# Step 2: Point correction (optional)
# ---------------------------------------------------------------------------
def point_correct(data_dir, json_dir, max_workers, kernel_size):
    rescale_dir = json_dir + "_rescale"
    ruler_dir = json_dir + "_ruler"
    gradient_dir = json_dir + "_gradient"
    cycle_dir = json_dir + "_gradient_cycle"
    restored_dir = json_dir + "_restored"

    try:
        print("  2a. Transform points to ruler coords")
        batch_transformer(json_dir, rescale_dir, max_workers=max_workers)

        print("  2b. Extract ruler regions")
        extract_ruler(data_dir, json_dir, ruler_dir, only_ruler=True, max_workers=max_workers)

        print("  2c. Compute gradient")
        os.makedirs(gradient_dir, exist_ok=True)
        ruler_imgs = [n for n in os.listdir(ruler_dir) if osp.splitext(n)[1].lower() in IMG_EXTS]
        if ruler_imgs:
            tasks = [(ruler_dir, gradient_dir, n, kernel_size) for n in ruler_imgs]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_gradient_one, tasks))
            print(f"    ok={sum(1 for ok, _, _ in results if ok)}")

        print("  2d. Nearest point correction")
        os.makedirs(cycle_dir, exist_ok=True)
        grad_imgs = [n for n in os.listdir(gradient_dir) if osp.splitext(n)[1].lower() in IMG_EXTS]
        if grad_imgs:
            tasks = [(gradient_dir, rescale_dir, cycle_dir, n) for n in grad_imgs]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_nearest_one, tasks))
            print(f"    ok={sum(1 for ok, _, _ in results if ok)}")

        print("  2e. Restore points to original coords")
        batch_restore(json_dir, rescale_dir, restored_dir, max_workers=max_workers)

        for fname in os.listdir(restored_dir):
            if fname.endswith(".json"):
                shutil.copy2(osp.join(restored_dir, fname), osp.join(json_dir, fname))

        print("[Point Correct] Done. JSONs updated.")

    finally:
        for d in [rescale_dir, ruler_dir, gradient_dir, cycle_dir, restored_dir]:
            if osp.exists(d):
                shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Step 3: Compute heights
# ---------------------------------------------------------------------------
def _height_one(image_path, label_path, img_name):
    """Compute plant height for a single image."""
    try:
        return compute_single_image(image_path, label_path, img_name)
    except Exception:
        return None


def compute_heights(data_dir, json_dir, max_workers):
    img_names = list_images(data_dir)
    func = partial(_height_one, data_dir, json_dir)

    heights = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for name, res in zip(img_names, ex.map(func, img_names)):
            if res is not None:
                heights.update(res)

    print(f"[Height] Computed for {len(heights)}/{len(img_names)} images")
    return heights


# ---------------------------------------------------------------------------
# Step 4: Visualize
# ---------------------------------------------------------------------------
def _draw_one(task):
    img_dir, json_dir, save_dir, img_name, height = task

    img = cv2.imread(osp.join(img_dir, img_name))
    if img is None:
        return False, img_name, "read failed"

    h, w = img.shape[:2]

    json_fp = osp.join(json_dir, osp.splitext(img_name)[0] + ".json")
    if osp.exists(json_fp):
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        thickness, radius, font_scale = 4, 8, 0.8
        for shp in data.get("shapes", []):
            stype = shp.get("shape_type", "")
            label = shp.get("label", "")
            pts = shp.get("points", [])

            if stype == "rectangle" and len(pts) == 2:
                x1 = max(0, min(int(round(pts[0][0])), w - 1))
                y1 = max(0, min(int(round(pts[0][1])), h - 1))
                x2 = max(0, min(int(round(pts[1][0])), w - 1))
                y2 = max(0, min(int(round(pts[1][1])), h - 1))
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), BBOX_COLOR, thickness)
                if label:
                    cv2.putText(img, label, (xmin, max(0, ymin - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, BBOX_COLOR, thickness, cv2.LINE_AA)

            elif stype == "point" and len(pts) == 1:
                x = max(0, min(int(round(pts[0][0])), w - 1))
                y = max(0, min(int(round(pts[0][1])), h - 1))
                cv2.circle(img, (x, y), radius, POINT_COLOR, -1)
                cv2.line(img, (x - radius * 2, y), (x + radius * 2, y), POINT_COLOR, max(1, thickness - 1))
                cv2.line(img, (x, y - radius * 2), (x, y + radius * 2), POINT_COLOR, max(1, thickness - 1))
                if label:
                    cv2.putText(img, label, (x + radius + 3, y - radius - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, POINT_COLOR, max(1, thickness - 1), cv2.LINE_AA)

    # Height banner
    rect_h, rect_w = int(h * 0.1), int(w * 0.3)
    cv2.rectangle(img, (0, 0), (rect_w, rect_h), HEIGHT_BG_COLOR, -1)
    scale = rect_h / 60 if rect_h > 0 else 1.0
    text = f"{height} cm"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 5)
    cv2.putText(img, text, (int((rect_w - tw) / 2), int((rect_h + th) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, HEIGHT_TEXT_COLOR, 5)

    ok = cv2.imwrite(osp.join(save_dir, img_name), img)
    return ok, img_name, "" if ok else "write failed"


def visualize(data_dir, json_dir, heights, save_img, max_workers):
    os.makedirs(save_img, exist_ok=True)

    tasks = [
        (data_dir, json_dir, save_img, name, heights[name])
        for name in list_images(data_dir) if name in heights
    ]
    if not tasks:
        print("[Visualize] No images to process")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_draw_one, tasks))

    ok_cnt = sum(1 for ok, _, _ in results if ok)
    print(f"[Visualize] Done. saved={ok_cnt} to {save_img}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--data", type=str, required=True, help="Image data directory")
    parser.add_argument("--save_json", type=str, required=True, help="Output directory for inference JSON results")
    parser.add_argument("--save_img", type=str, default=None, help="Output directory for visualization images (default: {save_json}_vis)")
    parser.add_argument("--point_correct", action="store_true", help="Enable point correction algorithm")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_det", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=3)
    args = parser.parse_args()

    data_dir = osp.abspath(args.data)
    json_dir = osp.abspath(args.save_json)
    save_img = osp.abspath(args.save_img) if args.save_img else json_dir + "_vis"

    # Step 1
    print("=" * 60)
    print("Step 1 / 4  YOLO Inference")
    print("=" * 60)
    inference(args.weight, data_dir, json_dir, args.device, args.imgsz, args.max_det)

    # Step 2 (optional)
    if args.point_correct:
        print("\n" + "=" * 60)
        print("Step 2 / 4  Point Correction")
        print("=" * 60)
        point_correct(data_dir, json_dir, args.max_workers, args.kernel_size)

    # Step 3
    print("\n" + "=" * 60)
    print("Step 3 / 4  Compute Plant Heights")
    print("=" * 60)
    heights = compute_heights(data_dir, json_dir, args.max_workers)
    height_json = osp.join(json_dir, "heights.json")
    with open(height_json, "w", encoding="utf-8") as f:
        json.dump(heights, f, indent=2, ensure_ascii=False)
    print(f"[Height] Saved to {height_json}")

    # Step 4
    print("\n" + "=" * 60)
    print("Step 4 / 4  Visualization")
    print("=" * 60)
    visualize(data_dir, json_dir, heights, save_img, args.max_workers)

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"  JSON:   {json_dir}")
    print(f"  Height: {height_json}")
    print(f"  Images: {save_img}")


if __name__ == "__main__":
    main()
