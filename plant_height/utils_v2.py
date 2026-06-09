"""Utility functions for plant height pipeline.

Consolidated from tools/{post_process, only_points, compute_height}.py.
All LocalBackend/fileio calls replaced with standard os/json/cv2 I/O.
"""

import io
import itertools
import json
import math
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Helpers (replaces ultralytics.fileio LocalBackend/JsonHandler)
# ---------------------------------------------------------------------------
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# From tools/compute_height.py
# ---------------------------------------------------------------------------
def calculate_distance(up_left, up_right, down_left, tip):
    if up_left[0] == down_left[0]:
        k = None
    else:
        k = (down_left[1] - up_left[1]) / (down_left[0] - up_left[0])

    if up_left[0] == up_right[0]:
        raise ValueError("up_left and up_right have the same x-coordinate")
    k1 = (up_right[1] - up_left[1]) / (up_right[0] - up_left[0])

    uldl_distance = math.sqrt((down_left[0] - up_left[0]) ** 2 + (down_left[1] - up_left[1]) ** 2)

    if k is None:
        x_cross = tip[0]
        y_cross = k1 * (x_cross - up_left[0]) + up_left[1]
        cross_point = [x_cross, y_cross]
    elif abs(k - k1) < 1e-10:
        raise ValueError("Lines are parallel, no intersection point")
    else:
        numerator = k * tip[0] - tip[1] - k1 * up_left[0] + up_left[1]
        denominator = k - k1
        x_cross = numerator / denominator
        y_cross = k * (x_cross - tip[0]) + tip[1]
        cross_point = [x_cross, y_cross]

    L = math.sqrt((cross_point[0] - tip[0]) ** 2 + (cross_point[1] - tip[1]) ** 2)
    return L, uldl_distance


def compute_single_image(image_path, label_path, image_name):
    img_path = osp.join(image_path, image_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    filename, _ = osp.splitext(image_name)
    label_name = filename + ".json"
    label_path_full = osp.join(label_path, label_name)
    if not osp.exists(label_path_full):
        raise FileNotFoundError(f"Label file not found: {label_path_full}")

    with open(label_path_full, "r") as f:
        labels = json.load(f)

    points = {}
    for vals in labels["shapes"]:
        if vals["label"] in ["up_left", "up_right", "down_left", "tip"]:
            points[vals["label"]] = vals["points"][0]

    required_points = ["up_left", "up_right", "down_left", "tip"]
    if not all(point in points for point in required_points):
        raise ValueError(f"Missing required points in {label_name}")

    L, uldl_distance = calculate_distance(
        points["up_left"],
        points["up_right"],
        points["down_left"],
        points["tip"],
    )

    height = L * 50 / uldl_distance
    tmp = round(height / 10, 4)
    tmp = f"{100 - tmp:.4f}"
    return {image_name: float(tmp)}


# ---------------------------------------------------------------------------
# From tools/only_points.py
# ---------------------------------------------------------------------------
def transform_points_in_bbox(json_path, output_path=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ruler_bbox = None
    for shape in data["shapes"]:
        if shape["label"] == "ruler" and shape["shape_type"] == "rectangle":
            ruler_bbox = shape
            break

    if not ruler_bbox:
        raise ValueError

    x1, y1 = ruler_bbox["points"][0]
    x2, y2 = ruler_bbox["points"][1]

    bbox_xmin = int(min(x1, x2))
    bbox_ymin = int(min(y1, y2))
    bbox_xmax = int(max(x1, x2))
    bbox_ymax = int(max(y1, y2))

    point_shapes = [s for s in data["shapes"] if s["shape_type"] == "point"]
    if not point_shapes:
        raise ValueError

    new_point_shapes = []
    for shape in point_shapes:
        orig_x, orig_y = shape["points"][0]
        new_x = orig_x - bbox_xmin
        new_y = orig_y - bbox_ymin
        new_shape = {
            "label": shape["label"],
            "points": [[new_x, new_y]],
            "group_id": shape["group_id"],
            "description": shape["description"],
            "shape_type": "point",
            "flags": shape.get("flags", {}),
            "mask": None,
        }
        new_point_shapes.append(new_shape)

    new_data = {
        "version": data["version"],
        "flags": data["flags"],
        "shapes": new_point_shapes,
        "imagePath": data["imagePath"],
        "imageHeight": int(bbox_ymax - bbox_ymin),
        "imageWidth": int(bbox_xmax - bbox_xmin),
    }

    if output_path is None:
        base_name = os.path.splitext(json_path)[0]
        output_path = f"{base_name}_transformed.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return new_data


def _find_ruler_bbox_xymin(data: dict) -> tuple[float, float]:
    ruler_bbox = None
    for shape in data.get("shapes", []):
        if shape.get("label") == "ruler" and shape.get("shape_type") == "rectangle":
            ruler_bbox = shape
            break
    if not ruler_bbox:
        raise ValueError

    (x1, y1), (x2, y2) = ruler_bbox["points"]
    bbox_xmin = int(min(x1, x2))
    bbox_ymin = int(min(y1, y2))
    return bbox_xmin, bbox_ymin


def restore_points_to_original(
    original_json_path: str,
    point_only_json_path: str,
    output_path: str | None = None,
    mode: str = "update_by_key",
    key_fields: tuple[str, ...] = ("label", "group_id"),
):
    with open(original_json_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    with open(point_only_json_path, "r", encoding="utf-8") as f:
        pts = json.load(f)

    bbox_xmin, bbox_ymin = _find_ruler_bbox_xymin(orig)

    restored_points = []
    for s in pts.get("shapes", []):
        if s.get("shape_type") != "point":
            continue
        ax, ay = s["points"][0]
        ox = ax + bbox_xmin
        oy = ay + bbox_ymin

        new_s = dict(s)
        new_s["points"] = [[ox, oy]]
        new_s["shape_type"] = "point"
        restored_points.append(new_s)

    if not restored_points:
        raise ValueError

    if mode == "replace_all_points":
        orig["shapes"] = [s for s in orig.get("shapes", []) if s.get("shape_type") != "point"]
        orig["shapes"].extend(restored_points)

    elif mode == "update_by_key":
        index = {}
        for s in orig.get("shapes", []):
            if s.get("shape_type") == "point":
                k = tuple(s.get(f) for f in key_fields)
                index[k] = s

        for s in restored_points:
            k = tuple(s.get(f) for f in key_fields)
            if k in index:
                index[k]["points"] = s["points"]
            else:
                orig.setdefault("shapes", []).append(s)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if output_path is None:
        base = os.path.splitext(original_json_path)[0]
        output_path = f"{base}_restored.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(orig, f, indent=2, ensure_ascii=False)

    return output_path


def _transform_one(task: tuple[str, str]):
    in_json, out_json = task
    try:
        transform_points_in_bbox(json_path=in_json, output_path=out_json)
        return True, os.path.basename(in_json), out_json
    except Exception as e:
        return False, os.path.basename(in_json), str(e)


def _restore_one(task: tuple[str, str, str], mode: str = "update_by_key"):
    original_json, point_only_json, out_json = task
    try:
        restore_points_to_original(
            original_json_path=original_json,
            point_only_json_path=point_only_json,
            output_path=out_json,
            mode=mode,
        )
        return True, os.path.basename(original_json), out_json
    except Exception as e:
        return False, os.path.basename(original_json), str(e)


def batch_transformer(json_path: str, output_path: str, max_workers: int = 8, suffix: str = ".json"):
    json_path = Path(osp.abspath(json_path))
    output_path = Path(osp.abspath(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_path.glob(f"*{suffix}")) if json_path.is_dir() else [json_path]
    json_files = [p for p in json_files if p.exists()]

    if not json_files:
        print(f"Not found to-transform json: {json_path}")
        return

    tasks = [(str(jp), str(output_path / jp.name)) for jp in json_files]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_transform_one, tasks))

    ok_cnt = sum(1 for ok, _, _ in results if ok)
    fail = [(name, err) for ok, name, err in results if not ok]
    print(f"Transform done. ok={ok_cnt}, failed={len(fail)}")
    if fail:
        print("Some failures (first 20):")
        for name, err in fail[:20]:
            print(f" - {name}: {err}")


def batch_restore(
    original_dir: str,
    point_only_dir: str,
    out_dir: str,
    max_workers: int = 8,
    suffix: str = ".json",
    mode: str = "update_by_key",
):
    original_dir = Path(osp.abspath(original_dir))
    point_only_dir = Path(osp.abspath(point_only_dir))
    out_dir = Path(osp.abspath(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    originals = sorted(original_dir.glob(f"*{suffix}"))
    if not originals:
        return

    tasks = []
    skipped = 0
    for oj in originals:
        pj = point_only_dir / oj.name
        if not pj.exists():
            skipped += 1
            continue
        out_path = out_dir / oj.name
        tasks.append((str(oj), str(pj), str(out_path)))

    if skipped:
        print(f"[Hint] {skipped} original json files with no matching point json.")

    if not tasks:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_restore_one, tasks, itertools.repeat(mode)))

    ok_cnt = sum(1 for ok, _, _ in results if ok)
    fail = [(name, err) for ok, name, err in results if not ok]
    print(f"Restore done. ok={ok_cnt}, failed={len(fail)}")
    if fail:
        print("Some failures (first 20):")
        for name, err in fail[:20]:
            print(f" - {name}: {err}")


# ---------------------------------------------------------------------------
# From tools/post_process.py
# ---------------------------------------------------------------------------
def compute_ruler_bbox(shapes, target_label: str = "ruler"):
    ruler = None
    for s in shapes or []:
        if s.get("label") == target_label:
            ruler = s
            break
    if ruler is None:
        raise ValueError(f"'{target_label}' not found")

    pts = ruler.get("points") or []
    if len(pts) < 2:
        raise ValueError(f"bad points for '{target_label}'")

    xs, ys = [], []
    for pt in pts:
        if len(pt) != 2:
            raise ValueError
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))

    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _extract_ruler_one(task: tuple[str, str, str, str, bool]):
    img_dir, label_dir, save_dir, img_name, only_ruler = task

    stem, ext = osp.splitext(img_name)
    if ext.lower() not in IMG_EXTS:
        return False, img_name, "skip non-image"

    json_fp = osp.join(label_dir, stem + ".json")
    if not osp.exists(json_fp):
        return False, img_name, f"label not found: {json_fp}"

    labels = _load_json(json_fp)
    if "shapes" not in labels:
        return False, img_name, "missing shapes"

    x_min, y_min, x_max, y_max = compute_ruler_bbox(labels["shapes"], target_label="ruler")

    image = cv2.imread(osp.join(img_dir, img_name), cv2.IMREAD_COLOR)
    if image is None:
        return False, img_name, f"imread failed: {osp.join(img_dir, img_name)}"
    h, w = image.shape[:2]

    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h))
    if not (x_max > x_min and y_max > y_min):
        return False, img_name, f"empty crop: ({x_min},{y_min})-({x_max},{y_max})"

    roi = image[y_min:y_max, x_min:x_max]
    out_fp = osp.join(save_dir, img_name)
    if only_ruler:
        cv2.imwrite(out_fp, roi)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y_min:y_max, x_min:x_max] = 255
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(out_fp, masked_image)

    return True, img_name, out_fp


def extract_ruler(img_pth, label_pth, save_pth, only_ruler, max_workers: int = 8):
    img_pth = osp.abspath(str(img_pth))
    label_pth = osp.abspath(str(label_pth))
    save_pth = osp.abspath(str(save_pth))
    os.makedirs(save_pth, exist_ok=True)

    img_list = [n for n in os.listdir(img_pth) if osp.splitext(n)[1].lower() in IMG_EXTS]
    if not img_list:
        raise ValueError(f"no images in {img_pth}")

    tasks = [(img_pth, label_pth, save_pth, n, bool(only_ruler)) for n in img_list]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_extract_ruler_one, tasks))

    ok_cnt = sum(1 for ok, _, _ in results if ok)
    failed = [(name, info) for ok, name, info in results if not ok and info != "skip non-image"]
    print(f"Extract ruler done. ok={ok_cnt}, failed={len(failed)}")
    if failed:
        print("Some failures (first 20):")
        for name, info in failed[:20]:
            print(f" - {name}: {info}")


def _gradient_one(task: tuple[str, str, str, int]):
    src_dir, save_dir, img_name, kernel_size = task

    ext = osp.splitext(img_name)[1].lower()
    if ext not in IMG_EXTS:
        return False, img_name, "skip non-image"

    img_fp = osp.join(src_dir, img_name)
    img = cv2.imread(img_fp, cv2.IMREAD_COLOR)
    if img is None:
        return False, img_name, f"imread failed: {img_fp}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=kernel_size)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    gradient_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, gradient_high_contrast = cv2.threshold(
        gradient_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    out_fp = osp.join(save_dir, img_name)
    cv2.imwrite(out_fp, gradient_high_contrast)
    return True, img_name, out_fp


def find_two_closest_points_in_circle(gray: np.ndarray, center, r: int):
    x0, y0 = center
    h, w = gray.shape[:2]

    yy, xx = np.ogrid[:h, :w]
    circle_mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= r * r
    ys, xs = np.where(circle_mask & (gray == 255))
    if xs.size < 2:
        raise ValueError

    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    order = np.argsort(d2)

    p1 = (int(xs[order[0]]), int(ys[order[0]]))
    p2 = None
    for idx in order[1:]:
        cand = (int(xs[idx]), int(ys[idx]))
        if cand != p1:
            p2 = cand
            break
    if p2 is None:
        raise ValueError

    return [p1, p2]


def centroid_if_triangle_else_fallback(p0, p1, p2, fallback=None, eps=0.0):
    if fallback is None:
        fallback = p0

    if len({p0, p1, p2}) < 3:
        return fallback

    area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
    if abs(area2) <= eps:
        return fallback

    cx = (p0[0] + p1[0] + p2[0]) / 3.0
    cy = (p0[1] + p1[1] + p2[1]) / 3.0
    return (int(round(cx)), int(round(cy)))


def update_point_in_labels(labels, label_name, new_pt):
    x, y = new_pt
    for shp in labels.get("shapes", []):
        if shp.get("label") == label_name:
            shp["points"] = [[float(x), float(y)]]
            return True
    return False


def _nearest_one(task: tuple[str, str, str, str]):
    img_dir, label_dir, save_dir, img_name = task

    stem, ext = osp.splitext(img_name)
    if ext.lower() not in IMG_EXTS:
        return False, img_name, "skip non-image"

    json_path = osp.join(label_dir, stem + ".json")
    if not osp.exists(json_path):
        return False, img_name, "skip missing json"

    labels = _load_json(json_path)
    os.makedirs(save_dir, exist_ok=True)

    pts_map = {}
    for shp in labels.get("shapes", []):
        lb = shp.get("label")
        if lb in ("up_left", "up_right", "down_left"):
            x = int(shp["points"][0][0])
            y = int(shp["points"][0][1])
            pts_map[lb] = (x, y)

    if not all(k in pts_map for k in ("up_left", "up_right", "down_left")):
        return False, img_name, "skip missing key points"

    bgr = cv2.imread(osp.join(img_dir, img_name), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, img_name, f"imread failed: {osp.join(img_dir, img_name)}"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _search_two_pts(center, base_r: int = 10, expand: int = 9):
        try:
            return find_two_closest_points_in_circle(gray, center, base_r)
        except Exception:
            for i in range(1, expand + 1):
                try:
                    return find_two_closest_points_in_circle(gray, center, base_r + i)
                except Exception:
                    continue
        raise ValueError("no two points found")

    r0 = 10
    try:
        pts_up_left = _search_two_pts(pts_map["up_left"], base_r=r0)
        pts_up_right = _search_two_pts(pts_map["up_right"], base_r=r0)
        pts_down_left = _search_two_pts(pts_map["down_left"], base_r=r0)
    except Exception as e:
        return False, img_name, f"skip search failed: {e}"

    centroid_up_left = centroid_if_triangle_else_fallback(pts_map["up_left"], pts_up_left[0], pts_up_left[1])
    centroid_up_right = centroid_if_triangle_else_fallback(pts_map["up_right"], pts_up_right[0], pts_up_right[1])
    centroid_down_left = centroid_if_triangle_else_fallback(pts_map["down_left"], pts_down_left[0], pts_down_left[1])

    # Draw on gradient image: blue=before, red=after
    vis = bgr.copy()
    corrected = {
        "up_left": centroid_up_left,
        "up_right": centroid_up_right,
        "down_left": centroid_down_left,
    }
    for label_name, orig in pts_map.items():
        corr = corrected[label_name]
        cv2.circle(vis, orig, 5, (255, 0, 0), -1)  # blue = before
        cv2.circle(vis, corr, 5, (0, 0, 255), -1)  # red = after
        cv2.line(vis, orig, corr, (0, 255, 0), 1)   # green line connecting
    cv2.imwrite(osp.join(save_dir, img_name), vis)

    update_point_in_labels(labels, "up_left", centroid_up_left)
    update_point_in_labels(labels, "up_right", centroid_up_right)
    update_point_in_labels(labels, "down_left", centroid_down_left)
    _dump_json(json_path, labels)

    return True, img_name, osp.join(save_dir, img_name)


def _refine_one(task: tuple[str, str, str, str]):
    """Refine keypoints using gradient-magnitude weighted centroid."""
    ruler_dir, label_dir, save_dir, img_name = task

    stem, ext = osp.splitext(img_name)
    if ext.lower() not in IMG_EXTS:
        return False, img_name, "skip non-image"

    json_path = osp.join(label_dir, stem + ".json")
    if not osp.exists(json_path):
        return False, img_name, "skip missing json"

    labels = _load_json(json_path)
    os.makedirs(save_dir, exist_ok=True)

    pts_map = {}
    for shp in labels.get("shapes", []):
        lb = shp.get("label")
        if lb in ("up_left", "up_right", "down_left"):
            pts_map[lb] = (float(shp["points"][0][0]), float(shp["points"][0][1]))

    if not all(k in pts_map for k in ("up_left", "up_right", "down_left")):
        return False, img_name, "skip missing key points"

    bgr = cv2.imread(osp.join(ruler_dir, img_name), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, img_name, f"imread failed: {osp.join(ruler_dir, img_name)}"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2)
    h, w = mag.shape

    search_radius = 8
    max_blend = 0.3

    def _refine_point(center, radius, max_bl, min_blend=0.03):
        cx, cy = center
        r = int(radius)

        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            return center

        y_lo = max(0, int(cy) - r)
        y_hi = min(h, int(cy) + r + 1)
        x_lo = max(0, int(cx) - r)
        x_hi = min(w, int(cx) + r + 1)
        if y_hi <= y_lo or x_hi <= x_lo:
            return center

        local_mag = mag[y_lo:y_hi, x_lo:x_hi]
        local_max = float(local_mag.max())
        if local_max < 1e-10:
            return center

        kx = min(max(int(round(cx)), 0), w - 1)
        ky = min(max(int(round(cy)), 0), h - 1)
        edge_ratio = float(mag[ky, kx]) / local_max
        deficit = max(0.0, 1.0 - edge_ratio)
        blend = max_bl * deficit * deficit
        if blend < min_blend:
            return center

        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        sigma = 1.0
        gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))

        thresh = np.percentile(local_mag, 80)
        mask = local_mag >= thresh

        weights = local_mag * gauss * mask
        total_w = weights.sum()
        if total_w < 1e-10:
            return center

        new_x = float((weights * xx).sum() / total_w)
        new_y = float((weights * yy).sum() / total_w)
        return (cx * (1 - blend) + new_x * blend, cy * (1 - blend) + new_y * blend)

    corrected = {}
    for label_name in ("up_left", "up_right", "down_left"):
        corrected[label_name] = _refine_point(pts_map[label_name], search_radius, max_blend)

    # For down_left: skip correction if displacement is too small (likely noise)
    dl_disp = math.sqrt(
        (corrected["down_left"][0] - pts_map["down_left"][0]) ** 2
        + (corrected["down_left"][1] - pts_map["down_left"][1]) ** 2
    )
    if dl_disp < 0.3:
        corrected["down_left"] = pts_map["down_left"]

    # Clamp individual keypoint displacement to limit outlier corrections
    max_allowed = 0.5
    for k in corrected:
        ox, oy = pts_map[k]
        cx, cy = corrected[k]
        dx, dy = cx - ox, cy - oy
        dist = math.sqrt(dx**2 + dy**2)
        if dist > max_allowed:
            scale = max_allowed / dist
            corrected[k] = (ox + dx * scale, oy + dy * scale)

    # Visualisation: blue=before, red=after, green line connecting
    vis = bgr.copy()
    for label_name, orig in pts_map.items():
        corr = corrected[label_name]
        o = (int(round(orig[0])), int(round(orig[1])))
        c = (int(round(corr[0])), int(round(corr[1])))
        cv2.circle(vis, o, 5, (255, 0, 0), -1)
        cv2.circle(vis, c, 5, (0, 0, 255), -1)
        cv2.line(vis, o, c, (0, 255, 0), 1)
    cv2.imwrite(osp.join(save_dir, img_name), vis)

    for label_name, new_pt in corrected.items():
        update_point_in_labels(labels, label_name, new_pt)
    _dump_json(json_path, labels)

    return True, img_name, osp.join(save_dir, img_name)
