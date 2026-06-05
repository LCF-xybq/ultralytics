"""train_val_test.py - Split dataset into train/val/test sets.

Usage:
    python train_val_test.py --input /path/to/data --save /path/to/output
    python train_val_test.py --input /path/to/data --save /path/to/output --ratio 7 2 1 --seed 42
"""

import argparse
import os
import os.path as osp
import random
import shutil


def split_dataset(input_dir, save_dir, ratios, seed):
    img_dir = osp.join(input_dir, "images")
    lbl_dir = osp.join(input_dir, "labels")
    assert osp.isdir(img_dir), f"images/ not found in {input_dir}"
    assert osp.isdir(lbl_dir), f"labels/ not found in {input_dir}"

    # Collect stems that have both image and label
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_stems = {}
    for f in os.listdir(img_dir):
        stem, ext = osp.splitext(f)
        if ext.lower() in img_exts:
            img_stems[stem] = f

    lbl_stems = set()
    for f in os.listdir(lbl_dir):
        stem, ext = osp.splitext(f)
        if ext.lower() == ".txt":
            lbl_stems.add(stem)

    valid_stems = sorted(set(img_stems) & lbl_stems)
    print(f"Found {len(valid_stems)} matched image-label pairs "
          f"({len(img_stems)} images, {len(lbl_stems)} labels)")

    random.seed(seed)
    random.shuffle(valid_stems)

    total = len(valid_stems)
    r_sum = sum(ratios)
    n_train = int(total * ratios[0] / r_sum)
    n_val = int(total * ratios[1] / r_sum)
    train_stems = valid_stems[:n_train]
    val_stems = valid_stems[n_train : n_train + n_val]
    test_stems = valid_stems[n_train + n_val :]
    splits = {"train": train_stems, "val": val_stems, "test": test_stems}

    for split_name, stems in splits.items():
        img_dst = osp.join(save_dir, "images", split_name)
        lbl_dst = osp.join(save_dir, "labels", split_name)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for stem in stems:
            img_name = img_stems[stem]
            lbl_name = stem + ".txt"
            shutil.copy2(osp.join(img_dir, img_name), osp.join(img_dst, img_name))
            shutil.copy2(osp.join(lbl_dir, lbl_name), osp.join(lbl_dst, lbl_name))

        print(f"  {split_name}: {len(stems)}")

    # Save splits info
    info_path = osp.join(save_dir, "splits.txt")
    with open(info_path, "w") as f:
        f.write(f"seed={seed}  ratio={ratios[0]}:{ratios[1]}:{ratios[2]}\n")
        for split_name, stems in splits.items():
            f.write(f"{split_name}: {len(stems)}\n")

    print(f"Saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Input dir with images/ and labels/")
    parser.add_argument("--save", type=str, required=True, help="Output dir")
    parser.add_argument("--ratio", type=int, nargs=3, default=[7, 2, 1], help="Train:val:test ratio (default: 7 2 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    split_dataset(args.input, args.save, args.ratio, args.seed)


if __name__ == "__main__":
    main()
