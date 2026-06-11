"""metrics.py - Plant height test-set evaluation.

Computes Box/Pose detection metrics (P/R/mAP50/mAP50-95) and plant height
regression metrics (MAE/RMSE/R2) by comparing prediction JSONs against
ground-truth JSONs (both in LabelMe format).

Usage:
    python metrics.py --input <pred_json_dir> --label <gt_json_dir> --save <output.csv>
"""

import argparse
import json
import math
import os
import os.path as osp

import numpy as np
import torch

from ultralytics.utils.metrics import ap_per_class

from utils_v3 import calculate_distance

KPT_NAMES = ["up_left", "up_right", "down_left", "tip"]
KPT_IDX = {name: i for i, name in enumerate(KPT_NAMES)}
N_KPT = len(KPT_NAMES)
SIGMA = np.ones(N_KPT, dtype=np.float32) / N_KPT
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _load_jsons(folder):
    """Load all JSON files in *folder*, keyed by stem."""
    mapping = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        stem = osp.splitext(fname)[0]
        with open(osp.join(folder, fname), "r", encoding="utf-8") as f:
            mapping[stem] = json.load(f)
    return mapping


def _parse_bbox(data):
    """Return xyxy pixel bbox from LabelMe ruler rectangle, or None."""
    for s in data.get("shapes", []):
        if s.get("label") == "ruler" and s.get("shape_type") == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def _parse_keypoints(data):
    """Return dict {name: [x, y]} for the 4 keypoints, or None if any missing."""
    pts = {}
    for s in data.get("shapes", []):
        if s.get("shape_type") == "point" and s.get("label") in KPT_IDX:
            pts[s["label"]] = s["points"][0]
    if all(k in pts for k in KPT_NAMES):
        return pts
    return None


def _bbox_iou_xyxy(box1, box2, eps=1e-7):
    """IoU of two xyxy boxes (list/array of 4 floats)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + eps)


def _compute_oks(gt_kpts, pred_kpts, bbox_area, sigma=SIGMA, eps=1e-7):
    """OKS between GT and Pred keypoints (dicts {name: [x,y]}).

    Per-keypoint OKS = exp(-d_i^2 / (2 * s^2 * k_i^2)), then averaged.
    Consistent with ultralytics kpt_iou implementation.
    """
    oks_sum = 0.0
    for idx, name in enumerate(KPT_NAMES):
        d2 = (gt_kpts[name][0] - pred_kpts[name][0]) ** 2 + (gt_kpts[name][1] - pred_kpts[name][1]) ** 2
        k = (2 * sigma[idx]) ** 2
        oks_sum += math.exp(-d2 / (k * (bbox_area + eps) * 2))
    return oks_sum / N_KPT


def _compute_height(kpts):
    """Compute plant height (cm) from keypoints dict."""
    L, uldl_distance = calculate_distance(
        kpts["up_left"], kpts["up_right"], kpts["down_left"], kpts["tip"]
    )
    height = L * 50 / uldl_distance
    return round(height / 10, 4)


def evaluate(input_dir, label_dir, save_path):
    gt_map = _load_jsons(label_dir)
    pred_map = _load_jsons(input_dir)

    tp_list, tp_p_list, conf_list, pred_cls_list, target_cls_list = [], [], [], [], []
    gt_heights, pred_heights = [], []
    missed = 0

    for stem, gt_data in sorted(gt_map.items()):
        gt_bbox = _parse_bbox(gt_data)
        gt_kpts = _parse_keypoints(gt_data)
        if gt_bbox is None or gt_kpts is None:
            continue

        target_cls_list.append(0)

        pred_data = pred_map.get(stem)
        if pred_data is None:
            missed += 1
            continue

        pred_bbox = _parse_bbox(pred_data)
        pred_kpts = _parse_keypoints(pred_data)
        if pred_bbox is None or pred_kpts is None:
            missed += 1
            continue

        # Box IoU → tp
        iou = _bbox_iou_xyxy(gt_bbox, pred_bbox)
        tp_row = np.array([iou >= t for t in IOU_THRESHOLDS])
        tp_list.append(tp_row)

        # OKS → tp_p
        bbox_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]) * 0.53
        oks = _compute_oks(gt_kpts, pred_kpts, bbox_area)
        tp_p_row = np.array([oks >= t for t in IOU_THRESHOLDS])
        tp_p_list.append(tp_p_row)

        conf_list.append(1.0)
        pred_cls_list.append(0)

        # Height
        try:
            gt_heights.append(_compute_height(gt_kpts))
            pred_heights.append(_compute_height(pred_kpts))
        except Exception:
            pass

    n_detected = len(tp_list)
    n_total = len(gt_map)
    if n_detected == 0:
        print("No valid matched pairs found.")
        return

    tp = np.stack(tp_list)
    tp_p = np.stack(tp_p_list)
    conf = np.array(conf_list)
    pred_cls = np.array(pred_cls_list)
    target_cls = np.array(target_cls_list)

    # Ensure target_cls covers all GT samples (including missed)
    # Missed samples have no prediction, so they only contribute to target_cls count

    # Box metrics
    _, _, p_b, r_b, _, ap_b, _, _, _, _, _, _ = ap_per_class(
        tp, conf, pred_cls, target_cls, prefix="Box"
    )

    # Pose metrics
    _, _, p_p, r_p, _, ap_p, _, _, _, _, _, _ = ap_per_class(
        tp_p, conf, pred_cls, target_cls, prefix="Pose"
    )

    # Extract scalar values (single class = index 0)
    results = {
        "precision(B)": float(p_b[0]),
        "recall(B)": float(r_b[0]),
        "mAP50(B)": float(ap_b[0, 0]),
        "mAP50-95(B)": float(ap_b[0].mean()),
        "precision(P)": float(p_p[0]),
        "recall(P)": float(r_p[0]),
        "mAP50(P)": float(ap_p[0, 0]),
        "mAP50-95(P)": float(ap_p[0].mean()),
    }

    # Height regression metrics
    if gt_heights and pred_heights:
        gt_h = np.array(gt_heights)
        pred_h = np.array(pred_heights)
        diff = pred_h - gt_h
        results["height_MAE"] = float(np.abs(diff).mean())
        results["height_RMSE"] = float(np.sqrt((diff ** 2).mean()))
        if len(gt_h) > 1:
            ss_res = ((gt_h - pred_h) ** 2).sum()
            ss_tot = ((gt_h - gt_h.mean()) ** 2).sum()
            results["height_R2"] = float(1 - ss_res / (ss_tot + 1e-16))
        else:
            results["height_R2"] = 0.0
    else:
        results["height_MAE"] = 0.0
        results["height_RMSE"] = 0.0
        results["height_R2"] = 0.0

    results["total_images"] = n_total
    results["detected_images"] = n_detected
    results["missed_images"] = missed + (n_total - len(target_cls_list))

    # Diagnostics: per-image IoU / OKS distribution
    ious = np.array([_bbox_iou_xyxy(_parse_bbox(gt_map[s]), _parse_bbox(pred_map[s]))
                     for s in sorted(gt_map) if s in pred_map
                     and _parse_bbox(gt_map[s]) and _parse_bbox(pred_map[s])])
    print(f"\n  Box IoU   — min={ious.min():.4f}  median={np.median(ious):.4f}  "
          f"mean={ious.mean():.4f}  max={ious.max():.4f}")

    oks_list = []
    for s in sorted(gt_map):
        if s not in pred_map:
            continue
        gt_k = _parse_keypoints(gt_map[s])
        pr_k = _parse_keypoints(pred_map[s])
        gt_b = _parse_bbox(gt_map[s])
        if gt_k and pr_k and gt_b:
            area = (gt_b[2] - gt_b[0]) * (gt_b[3] - gt_b[1]) * 0.53
            oks_list.append(_compute_oks(gt_k, pr_k, area))
    if oks_list:
        oks_arr = np.array(oks_list)
        print(f"  OKS       — min={oks_arr.min():.4f}  median={np.median(oks_arr):.4f}  "
              f"mean={oks_arr.mean():.4f}  max={oks_arr.max():.4f}")

    # Height error distribution
    if gt_heights and pred_heights:
        diff = np.array(pred_heights) - np.array(gt_heights)
        print(f"  Height err— min={diff.min():.2f}  median={np.median(diff):.2f}  "
              f"mean={diff.mean():.2f}  max={diff.max():.2f} cm")

    # Save CSV
    os.makedirs(osp.dirname(osp.abspath(save_path)), exist_ok=True)
    header = ",".join(results.keys())
    values = ",".join(str(v) for v in results.values())
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(values + "\n")

    print(f"\n[Metrics] Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plant height test-set metrics")
    parser.add_argument("--input", type=str, required=True, help="Prediction JSON folder")
    parser.add_argument("--label", type=str, required=True, help="Ground-truth JSON folder")
    parser.add_argument("--save", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()
    evaluate(args.input, args.label, args.save)


if __name__ == "__main__":
    main()
