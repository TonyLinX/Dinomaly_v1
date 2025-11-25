#!/usr/bin/env python3
"""
Generate side-by-side images showing original (left) vs illumination-normalized (right).
Source images are taken from data/mvtec_ad_2/<class>/test_public/bad.
Outputs are saved to plots/normalize/<class>/<filename>_compare.png.
"""
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image

# Ensure project root is on sys.path so we can import preprocessing.py when run via `python tools/...`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import my_preprocessing


CLASSES: list[str] = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]

SOURCE_ROOT = Path("data/mvtec_ad_2")
OUTPUT_ROOT = Path("plots/normalize")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_side_by_side(original: Image.Image, normalized: Image.Image) -> Image.Image:
    """Place original on the left and normalized on the right."""
    width, height = original.size
    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(original, (0, 0))
    canvas.paste(normalized, (width, 0))
    return canvas


def process_class(class_name: str) -> None:
    src_dir = SOURCE_ROOT / class_name / "test_public" / "bad"
    if not src_dir.is_dir():
        print(f"[skip] {class_name}: folder not found ({src_dir})")
        return

    out_dir = OUTPUT_ROOT / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in iter_images(src_dir):
        original = Image.open(img_path).convert("RGB")
        normalized = my_preprocessing(original)
        combined = build_side_by_side(original, normalized)

        out_path = out_dir / f"{img_path.stem}_compare.png"
        combined.save(out_path)
        print(f"[ok] {class_name}: {out_path}")


def main() -> None:
    for cls in CLASSES:
        process_class(cls)


if __name__ == "__main__":
    main()
