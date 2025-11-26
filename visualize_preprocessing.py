#!/usr/bin/env python3
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocessing import my_preprocessing


def save_histogram(img: np.ndarray, path: Path, title: str) -> None:
    """Save per-channel histogram."""
    plt.figure(figsize=(6, 4))
    colors = ("r", "g", "b")
    for idx, color in enumerate(colors):
        plt.hist(
            img[..., idx].ravel(),
            bins=256,
            range=(0, 255),
            color=color,
            alpha=0.5,
            label=f"{color.upper()} channel",
        )
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_heatmap(arr: np.ndarray, path: Path, cmap: str, vmin=None, vmax=None) -> None:
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def main() -> None:
    input_path = Path("data/mvtec_ad_2/can/test_public/bad/000_shift_2.png")
    output_dir = Path("plots/my_preprocessing_viz")
    output_dir.mkdir(parents=True, exist_ok=True)

    pil_img = Image.open(input_path).convert("RGB")
    img_np = np.array(pil_img)

    blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=15, sigmaY=15)
    img_float = img_np.astype(np.float32)
    blur_float = blur.astype(np.float32)
    epsilon = 1.0
    ratio = img_float / (blur_float + epsilon)
    normalized_np = np.clip(ratio * 128.0, 0, 255).astype(np.uint8)

    # Keep consistency with the actual preprocessing entry point.
    normalized_func = np.array(my_preprocessing(pil_img))
    if not np.array_equal(normalized_np, normalized_func):
        raise RuntimeError("Local computation differs from my_preprocessing() output.")

    Image.fromarray(img_np).save(output_dir / "orig.png")
    Image.fromarray(blur).save(output_dir / "illum_blur.png")

    blur_gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    save_heatmap(blur_gray, output_dir / "illum_blur_gray.png", cmap="magma", vmin=0, vmax=255)

    ratio_vis = np.clip(ratio, 0, 2) / 2.0  # 0–1 range for imsave
    save_heatmap(ratio_vis, output_dir / "ratio_orig_over_blur.png", cmap="viridis", vmin=0, vmax=1)

    Image.fromarray(normalized_np).save(output_dir / "normalized.png")

    diff = img_np.astype(np.int16) - blur.astype(np.int16)
    diff_shifted = np.clip(diff + 128, 0, 255).astype(np.uint8)
    Image.fromarray(diff_shifted).save(output_dir / "diff_orig_minus_blur.png")

    save_histogram(img_np, output_dir / "hist_orig.png", "Original histogram")
    save_histogram(normalized_np, output_dir / "hist_normalized.png", "Normalized histogram")

    print(f"Saved visualization artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
