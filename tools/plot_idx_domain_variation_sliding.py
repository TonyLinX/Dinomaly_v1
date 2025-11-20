import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import matplotlib

matplotlib.use("Agg")

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from plot_idx_domain_variation import (  # noqa: E402
    ALLOWED_EXTS,
    build_encoder_model,
    compute_embeddings,
    list_images,
    parse_groups,
    parse_idx_domain,
    parse_int_list,
    plot_idx_domain_embedding,
    set_seed,
)


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def _grid_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        raise ValueError(f"Dimension {length} is smaller than patch_size={patch_size}.")
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


class SlidingWindowDataset(Dataset):
    def __init__(self, root: str, patch_size: int, stride: int, transform):
        self.transform = transform
        self.patch_size = patch_size
        self.samples: List[Tuple[str, str, str, Tuple[int, int, int, int]]] = []

        for path in list_images(root):
            idx, domain = parse_idx_domain(path)
            with Image.open(path) as img:
                width, height = img.size
            xs = _grid_positions(width, patch_size, stride)
            ys = _grid_positions(height, patch_size, stride)
            for top in ys:
                for left in xs:
                    box = (left, top, left + patch_size, top + patch_size)
                    self.samples.append((path, idx, domain, box))

        if not self.samples:
            raise RuntimeError(f"No patches could be constructed under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, img_idx, domain, box = self.samples[idx]
        with Image.open(path).convert("RGB") as img:
            patch = img.crop(box)
        if self.transform is not None:
            patch = self.transform(patch)
        return patch, img_idx, domain, path


def build_patch_transform(crop_size: int, mean: Sequence[float], std: Sequence[float]):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def extract_average_features(model, loader, device):
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    meta: Dict[str, Dict[str, str]] = {}

    with torch.no_grad():
        for imgs, idxs, domains, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            en, _ = model(imgs)
            pooled = [f.mean(dim=[2, 3]) for f in en]
            vec = torch.cat(pooled, dim=1).cpu().numpy()
            for i, path in enumerate(paths):
                if path not in sums:
                    sums[path] = np.zeros(vec.shape[1], dtype=np.float64)
                    counts[path] = 0
                    meta[path] = {"idx": idxs[i], "domain": domains[i], "path": path}
                sums[path] += vec[i]
                counts[path] += 1

    if not sums:
        raise RuntimeError("Feature extraction produced no outputs.")

    ordered_paths = sorted(meta.keys(), key=lambda p: (meta[p]["idx"], meta[p]["domain"], p))
    features = []
    meta_list = []
    for path in ordered_paths:
        features.append(sums[path] / max(counts[path], 1))
        meta_list.append(meta[path])
    return np.stack(features).astype(np.float32), meta_list


def main():
    parser = argparse.ArgumentParser(
        description="Sliding-window variant of idx-domain plotting with PCA & t-SNE."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/mvtec_ad_2/can/test_public/bad",
        help="Folder with <idx>_<domain>.png images.",
    )
    parser.add_argument("--output_dir", type=str, default="./domain_shift_idx_plots_sliding")
    parser.add_argument("--encoder_name", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=448,
        help="Spatial size of each extracted patch (no pre-resize).",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=406,
        help="Sliding stride between patches.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=392,
        help="Center-crop size applied to each 448x448 patch before tensor conversion.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=2000)
    parser.add_argument("--tsne_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"])
    parser.add_argument("--tsne_exaggeration", type=float, default=12.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_angle", type=float, default=0.5)
    parser.add_argument("--no_pca_tsne", action="store_true")
    parser.add_argument("--l2norm_before_tsne", action="store_true")
    parser.add_argument("--center_domain", type=str, default="regular")
    parser.add_argument(
        "--target_layers",
        type=str,
        default="2,3,4,5,6,7,8,9",
        help="Comma list of encoder block indices used for feature fusion.",
    )
    parser.add_argument(
        "--fuse_groups",
        type=str,
        default="0 1 2 3|4 5 6 7",
        help="Groups (space/comma separated) of layer indices fused together, separated by '|'.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for feature extraction.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1)
    device = torch.device(args.device)

    target_layers = parse_int_list(args.target_layers)
    fuse_groups = parse_groups(args.fuse_groups)

    transform = build_patch_transform(args.crop_size, DEFAULT_MEAN, DEFAULT_STD)
    dataset = SlidingWindowDataset(
        args.image_dir,
        patch_size=args.patch_size,
        stride=args.patch_stride,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_encoder_model(
        args.encoder_name,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_groups,
        device=device,
    )

    features, meta = extract_average_features(model, loader, device)
    np.save(os.path.join(args.output_dir, "features.npy"), features)

    embed_pca, embed_tsne = compute_embeddings(
        features=features,
        metric=args.tsne_metric,
        perplexity=args.perplexity,
        n_iter=args.tsne_iter,
        init=args.tsne_init,
        exaggeration=args.tsne_exaggeration,
        lr=args.tsne_lr,
        angle=args.tsne_angle,
        l2norm_before_tsne=args.l2norm_before_tsne,
        no_pca_tsne=args.no_pca_tsne,
    )

    plot_idx_domain_embedding(
        embed_pca,
        meta,
        os.path.join(args.output_dir, "pca_idx_domains.png"),
        "PCA (2D)",
        args.center_domain,
    )
    plot_idx_domain_embedding(
        embed_tsne,
        meta,
        os.path.join(args.output_dir, "tsne_idx_domains.png"),
        "t-SNE (2D)",
        args.center_domain,
    )

    print("Saved:")
    print(" -", os.path.join(args.output_dir, "features.npy"))
    print(" -", os.path.join(args.output_dir, "pca_idx_domains.png"))
    print(" -", os.path.join(args.output_dir, "tsne_idx_domains.png"))


if __name__ == "__main__":
    main()
