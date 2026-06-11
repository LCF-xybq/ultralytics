import argparse
import os

import cv2
import numpy as np
from scipy.fft import dctn, idctn


def high_pass_filter(shape, cutoff_ratio=0.1):
    """Generate a high-pass filter mask based on cutoff ratio.

    Args:
        shape: 2D shape (h, w) of the filter.
        cutoff_ratio: Ratio of low-frequency components to suppress (0~1).
    """
    h, w = shape
    mask = np.ones(shape, dtype=np.float32)
    ch = int(h * cutoff_ratio)
    cw = int(w * cutoff_ratio)
    mask[:ch, :cw] = 0
    return mask


def apply_dct_high_pass(image, cutoff_ratio=0.1):
    """Apply DCT-based high-pass filter to an image.

    Args:
        image: Input image (BGR, uint8).
        cutoff_ratio: Ratio of low-frequency components to suppress (0~1).

    Returns:
        result: Filtered image (BGR, uint8).
        freq: DCT spectrum (float32) for visualization.
    """
    # Convert to float32 YCrCb, process Y channel only
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb[:, :, 0]

    # DCT -> high-pass filter -> iDCT
    freq = dctn(y_channel, type=2, norm="ortho")
    mask = high_pass_filter(freq.shape, cutoff_ratio)
    filtered = idctn(freq * mask, type=2, norm="ortho")

    ycrcb[:, :, 0] = np.clip(filtered, 0, 255)
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR), freq


def save_dct_spectrum(freq, path):
    """Save DCT spectrum as a log-scaled grayscale image."""
    magnitude = np.log1p(np.abs(freq))
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, normalized)


def main():
    parser = argparse.ArgumentParser(description="DCT high-pass filter for image enhancement")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--cutoff", type=float, default=0.1, help="Low-frequency cutoff ratio (0~1, default: 0.1)")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    result, freq = apply_dct_high_pass(image, args.cutoff)
    cv2.imwrite(args.output, result)
    print(f"Saved high-pass filtered image to {args.output}")

    # Save DCT spectrum
    base, ext = os.path.splitext(args.output)
    dct_path = f"{base}_dct.png"
    save_dct_spectrum(freq, dct_path)
    print(f"Saved DCT spectrum to {dct_path}")


if __name__ == "__main__":
    main()
