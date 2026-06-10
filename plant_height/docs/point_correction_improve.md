# Point Correction Algorithm Improvement (v3)

## Problem

The v3 `_refine_one` gradient-magnitude weighted centroid algorithm had nearly no effect on keypoint correction. Points barely moved from their original positions, and metrics showed no meaningful improvement:

- V2 result: MAE 3.21→3.22, RMSE 4.42→4.40, R² 0.914→0.915

## Root Cause Analysis

| Parameter | Before | Problem |
|-----------|--------|---------|
| `search_radius` | 8 | Too small for ruler crops averaging 542x544px |
| `max_blend` | 0.3 | Combined with `deficit²`, actual blend usually < 0.1 |
| `sigma` | 1.0 | Gaussian too narrow, negligible neighborhood influence |
| `percentile` | 80 | Too high, too few valid edge pixels |
| `max_allowed` | 0.5 | **Critical**: 0.5px cap meaningless on 500+px images |
| `blend` formula | `max_bl * deficit²` | Quadratic decay overly conservative |

Additionally, partial correction (only some keypoints corrected while others stayed at original positions) broke geometric consistency between up_left, up_right, and down_left, causing height calculation errors to increase.

## Changes

File: `utils_v3.py` — `_refine_one` function

### 1. Expanded Search Range
- `search_radius`: 8 → **20**
- `sigma`: 1.0 → **3.0**
- `percentile` threshold: 80 → **60**

### 2. Adaptive Skip for Already-Accurate Points
- Skip correction when `edge_ratio > 0.7` (point is already on a strong gradient edge)

### 3. All-or-Nothing Correction Strategy
- Compute correction targets for all 3 keypoints (up_left, up_right, down_left)
- **Only apply corrections when all 3 points have valid targets**
- If any point fails validation, all points remain at original positions
- Prevents partial correction from breaking geometric relationships

### 4. Gradient Consistency Check
- Correction target must land on a higher-gradient region than the original position
- Rejects corrections that would move points away from edges

### 5. Linear Blend with Reasonable Displacement
- Blend formula: `max_bl * deficit` (linear, replacing quadratic `deficit²`)
- `max_blend`: 0.3 → **0.5**
- Displacement cap: 0.5px → **8px**
- Removed per-keypoint down_left special-case filter (handled by all-or-nothing)

## Results

All three models show improvement across all three metrics:

| Model | Version | MAE↓ | RMSE↓ | R²↑ |
|-------|---------|------|-------|-----|
| YOLOv8 | raw | 3.2080 | 4.4197 | 0.9138 |
| | corrected | **3.1885** | **4.3958** | **0.9148** |
| YOLO11 | raw | 3.2665 | 4.2790 | 0.9192 |
| | corrected | **3.1957** | **4.2529** | **0.9202** |
| YOLO26 | raw | 3.0013 | 3.8492 | 0.9346 |
| | corrected | **2.9850** | **3.8274** | **0.9354** |
