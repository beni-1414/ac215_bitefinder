# augment_images.py
# purpose: expand a bug bite image dataset on disk by writing augmented copies
# input: a root folder that contains class subfolders (e.g., training/bug_1, training/bug_2, ...)
# output: a mirrored folder under --output-dir with new files like img_xyz_aug001.jpg, img_xyz_aug002.jpg, ...

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import Iterable, Tuple, Optional

from PIL import Image
import torch
from torchvision import transforms as T


# ---------------------------
# helpers
# ---------------------------

def list_image_files(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    """yield all image files with allowed extensions under root/*/* (class subfolders)."""
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for fp in class_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                yield fp


def stable_int(s: str) -> int:
    """map any string to a stable positive int; useful for reproducible per-file seeds."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def ensure_rgb(img: Image.Image) -> Image.Image:
    """convert to rgb if needed so downstream torchvision ops behave consistently."""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def build_augmentations(
    image_size: Optional[int],
    strength: str,
    enable_perspective: bool = True,
    enable_blur: bool = True,
) -> T.Compose:
    """
    build a torchvision transform pipeline that outputs a pil image.
    image_size: if set, apply a random resized crop to roughly standardize scale and aspect.
    strength: controls magnitudes for color jitter and geometry ('light' | 'medium' | 'strong').
    enable_perspective / enable_blur: quick toggles for those ops.
    """
    # choose magnitudes based on requested strength
    if strength == "light":
        cj = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
        rot = 10
        persp = 0.15
        blur_k = 3
    elif strength == "strong":
        cj = dict(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.08)
        rot = 25
        persp = 0.35
        blur_k = 5
    else:
        # default to medium
        cj = dict(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05)
        rot = 15
        persp = 0.25
        blur_k = 3

    ops = []
    # do a content-preserving random crop + resize if a target size is given
    if image_size is not None:
        # scale and ratio ranges keep sensible crops while adding variety
        ops.append(T.RandomResizedCrop(size=image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)))

    # simple geometric transforms for viewpoint and reflection invariance
    ops.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomRotation(degrees=rot, expand=False),
    ])

    # perspective warp adds mild projective distortion (disabled via flag if desired)
    if enable_perspective:
        ops.append(T.RandomPerspective(distortion_scale=persp, p=0.25))

    # photometric transforms for lighting and camera variability
    ops.extend([
        T.RandomApply([T.ColorJitter(**cj)], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=1.5)], p=0.3),
    ])

    # simulate focus differences with blur (disabled via flag if desired)
    if enable_blur:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=blur_k)], p=0.25))

    return T.Compose(ops)


def make_output_path(in_fp: Path, out_root: Path, copy_idx: int, output_ext: str) -> Path:
    """
    mirror class subfolders and name augmented files deterministically.
    example:
      data/train/bug_1/img.jpg -> out/train/bug_1/img_aug001.jpg
    """
    # take 'class_dir/file' relative to the class root (two parents up: class_dir and its parent)
    rel = in_fp.relative_to(in_fp.parents[1])  # class_dir/file
    class_dir = rel.parent
    stem = in_fp.stem
    # use a 3-digit suffix for stable sorting and easy counting
    fname = f"{stem}_aug{copy_idx:03d}{output_ext}"
    return out_root / class_dir / fname


def save_pil(img: Image.Image, path: Path, quality: int = 95) -> None:
    """create parent dirs and save a pil image with reasonable defaults for jpg and png."""
    # ensure the parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # choose format-specific options
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        img.save(path, format="JPEG", quality=quality, optimize=True)
    elif path.suffix.lower() == ".png":
        img.save(path, format="PNG", compress_level=6, optimize=True)
    else:
        # if an unknown extension is passed, default to png to avoid surprises
        img.save(path.with_suffix(".png"), format="PNG", compress_level=6, optimize=True)


# ---------------------------
# main routine
# ---------------------------

def run(
    input_dir: str,
    output_dir: str,
    copies: int,
    image_size: Optional[int],
    strength: str,
    include_original: bool,
    exts: Tuple[str, ...],
    output_ext: str,
    seed: Optional[int],
    dry_run: bool,
) -> None:
    """
    walk input_dir, augment images, and write to output_dir while mirroring the class structure.
    copies: number of augmented versions to create per original image.
    include_original: if true, also copy the original image to the output tree (with aug index 000).
    seed: global seed that anchors randomness; per-copy seeds are derived from it and the file path.
    dry_run: if true, do not write files; only print progress and decisions.
    """
    # normalize paths and avoid surprises with relative working dirs
    in_root = Path(input_dir).resolve()
    out_root = Path(output_dir).resolve()

    # set global seeds if provided (per-copy seeds will be derived inside the loop)
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # build the augmentation pipeline once
    tfm = build_augmentations(image_size=image_size, strength=strength)

    # collect all eligible files under the input root
    files = list(list_image_files(in_root, exts))
    if not files:
        print("no input images found. check --input-dir and --exts")
        return

    print(f"found {len(files)} images. writing to {out_root}")
    for i, fp in enumerate(files, 1):
        # open and coerce to rgb so transforms receive a consistent mode
        try:
            img = Image.open(fp)
            img = ensure_rgb(img)
        except Exception as e:
            # skip any file that pillow cannot decode
            print(f"[skip] failed to open {fp}: {e}")
            continue

        # optionally copy the original image (index 0) so downstream stages can include it
        if include_original:
            orig_out = make_output_path(fp, out_root, 0, output_ext)
            if not dry_run:
                save_pil(img, orig_out)

        # generate n augmented copies per original
        for k in range(1, copies + 1):
            # build a stable per-copy seed so the same command reproduces the same outputs
            base = seed if seed is not None else 0
            derived = (stable_int(str(fp)) + 131 * k + 31 * base) & 0x7FFFFFFF
            random.seed(derived)
            torch.manual_seed(derived)

            try:
                # apply the stochastic pipeline to produce a new pil image
                aug_img = tfm(img)
                out_fp = make_output_path(fp, out_root, k, output_ext)
                if not dry_run:
                    save_pil(aug_img, out_fp)
            except Exception as e:
                # do not halt the run if a single image fails; just report and continue
                print(f"[skip] failed to augment {fp} copy {k}: {e}")

        # print periodic progress so long runs show liveness
        if i % 100 == 0:
            print(f"processed {i}/{len(files)}")

    print("done.")


def parse_args():
    # define a simple cli so teammates can run the same script with standard flags
    p = argparse.ArgumentParser(description="offline image augmentation to expand a kaggle-style folder dataset")
    p.add_argument("--input-dir", required=True, help="root dir containing class subfolders with images")
    p.add_argument("--output-dir", required=True, help="root dir to write augmented dataset (mirrors class layout)")
    p.add_argument("--copies", type=int, default=3, help="number of augmented copies per original image")
    p.add_argument("--image-size", type=int, default=None, help="optional square size for randomresizedcrop")
    p.add_argument("--strength", choices=["light", "medium", "strong"], default="medium", help="augmentation magnitude")
    p.add_argument("--include-original", action="store_true", help="also copy the original image to the output set")
    p.add_argument("--exts", default=".jpg,.jpeg,.png", help="comma-separated list of allowed input extensions")
    p.add_argument("--output-ext", choices=[".jpg", ".jpeg", ".png"], default=".jpg", help="extension to use when saving")
    p.add_argument("--seed", type=int, default=42, help="global seed for reproducibility")
    p.add_argument("--dry-run", action="store_true", help="do not write files, just print what would happen")
    return p.parse_args()


if __name__ == "__main__":
    # parse arguments and convert a few into normalized forms
    args = parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copies=args.copies,
        image_size=args.image_size,
        strength=args.strength,
        include_original=args.include_original,
        exts=tuple(
            # normalize extensions to include a leading dot and be lowercase
            s.strip().lower() if s.strip().startswith(".") else f".{s.strip().lower()}"
            for s in args.exts.split(",")
        ),
        output_ext=args.output_ext.lower(),
        seed=args.seed,
        dry_run=args.dry_run,
    )
