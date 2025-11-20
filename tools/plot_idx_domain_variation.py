import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import get_data_transforms  # noqa: E402
from models import vit_encoder  # noqa: E402
from models.uad import ViTill  # noqa: E402


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPEG", ".JPG"}


def set_seed(seed: int = 1) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")


def list_images(root: str) -> List[str]:
    root_path = Path(root)
    files: List[str] = []
    for ext in ALLOWED_EXTS:
        files.extend(str(p) for p in root_path.rglob(f"*{ext}"))
    return sorted({os.path.abspath(p) for p in files})


def parse_idx_domain(path: str) -> Tuple[str, str]:
    stem = Path(path).stem
    if "_" not in stem:
        raise ValueError(f"filename '{stem}' does not follow <idx>_<domain> convention")
    idx, remainder = stem.split("_", 1)
    if not idx:
        raise ValueError(f"missing idx prefix in '{stem}'")
    if not remainder:
        raise ValueError(f"missing domain suffix in '{stem}'")
    return idx, remainder


class IndexedDomainDataset(Dataset):
    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples: List[Tuple[str, str, str]] = []
        skipped: List[str] = []
        for path in list_images(root):
            try:
                idx, domain = parse_idx_domain(path)
            except ValueError:
                skipped.append(path)
                continue
            self.samples.append((path, idx, domain))
        if not self.samples:
            msg = f"No valid images found under {root}"
            if skipped:
                msg += f" (skipped {len(skipped)} malformed names)"
            raise RuntimeError(msg)
        if skipped:
            print(f"[WARN] Ignored {len(skipped)} files that do not match <idx>_<domain>.* naming.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, im_idx, domain = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, im_idx, domain, path


def build_encoder_model(
    encoder_name: str,
    target_layers: Sequence[int],
    fuse_layer_encoder: Sequence[Sequence[int]],
    device: torch.device,
) -> ViTill:
    encoder = vit_encoder.load(encoder_name)

    if "small" in encoder_name:
        embed_dim, num_heads = 384, 6
    elif "base" in encoder_name:
        embed_dim, num_heads = 768, 12
    elif "large" in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("encoder_name must include small/base/large")

    del embed_dim, num_heads  # kept for parity with training scripts

    bottleneck = nn.ModuleList([])
    decoder = nn.ModuleList([])
    fuse_layer_decoder: List[List[int]] = []

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=list(target_layers),
        fuse_layer_encoder=[list(g) for g in fuse_layer_encoder],
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0,
        remove_class_token=False,
    ).to(device)
    model.eval()
    return model


def extract_features_with_meta(model: ViTill, loader: DataLoader, device: torch.device):
    feats: List[torch.Tensor] = []
    meta: List[Dict[str, str]] = []
    with torch.no_grad():
        for imgs, idxs, domains, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            en, _ = model(imgs)
            pooled = [f.mean(dim=[2, 3]) for f in en]
            vec = torch.cat(pooled, dim=1).cpu()
            feats.append(vec)
            for i in range(len(paths)):
                meta.append({"idx": idxs[i], "domain": domains[i], "path": paths[i]})
    features = torch.cat(feats, dim=0).numpy()
    return features, meta


def compute_embeddings(
    features: np.ndarray,
    metric: str,
    perplexity: float,
    n_iter: int,
    init: str,
    exaggeration: float,
    lr: float,
    angle: float,
    l2norm_before_tsne: bool,
    no_pca_tsne: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    pca2 = PCA(n_components=2, random_state=0)
    embed_pca = pca2.fit_transform(features_std)

    Xt = features.copy()
    if l2norm_before_tsne or metric == "cosine":
        norms = np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
        Xt = Xt / norms
    else:
        Xt = features_std

    if no_pca_tsne:
        Xtsne_in = Xt
    else:
        pca_k = min(50, Xt.shape[1], len(Xt))
        pca_k = max(2, pca_k)
        pca_red = PCA(n_components=pca_k, random_state=0)
        Xtsne_in = pca_red.fit_transform(Xt)

    max_perp = max(5.0, (len(Xtsne_in) - 1) / 3.0)
    if perplexity >= max_perp:
        warnings.warn(
            f"perplexity {perplexity} is too high for {len(Xtsne_in)} samples; using {max_perp:.2f}"
        )
        perplexity = min(max_perp - 1e-3, perplexity)
        perplexity = max(5.0, perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        n_iter=n_iter,
        init=init,
        learning_rate=lr,
        early_exaggeration=exaggeration,
        angle=angle,
        random_state=0,
    )
    embed_tsne = tsne.fit_transform(Xtsne_in)
    return embed_pca, embed_tsne


def plot_idx_domain_embedding(
    embedding: np.ndarray,
    meta: Sequence[Dict[str, str]],
    out_path: str,
    title: str,
    center_domain: str,
) -> None:
    domains = sorted(
        {sample["domain"] for sample in meta},
        key=lambda d: (0 if d == center_domain else 1, d),
    )
    cmap = plt.get_cmap("tab20", len(domains))
    domain_colors = {dom: cmap(i) for i, dom in enumerate(domains)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for dom in domains:
        idxs = [i for i, sample in enumerate(meta) if sample["domain"] == dom]
        pts = embedding[idxs]
        if len(pts) == 0:
            continue
        marker = "X" if dom == center_domain else "o"
        size = 70 if dom == center_domain else 35
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=dom,
            color=[domain_colors[dom]],
            s=size,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.4,
            marker=marker,
        )

    by_idx: Dict[str, List[int]] = defaultdict(list)
    for i, sample in enumerate(meta):
        by_idx[sample["idx"]].append(i)

    missing_center: List[str] = []
    for idx_value, sample_indices in by_idx.items():
        center_idx = next(
            (
                si
                for si in sample_indices
                if meta[si]["domain"] == center_domain
            ),
            None,
        )
        if center_idx is None:
            missing_center.append(idx_value)
            continue
        cx, cy = embedding[center_idx]
        ax.text(
            cx,
            cy,
            idx_value,
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            color="black",
        )
        for si in sample_indices:
            if si == center_idx:
                continue
            sx, sy = embedding[si]
            dom = meta[si]["domain"]
            ax.plot(
                [cx, sx],
                [cy, sy],
                color=domain_colors.get(dom, "gray"),
                linewidth=0.8,
                alpha=0.8,
            )

    if missing_center:
        print(
            f"[WARN] center domain '{center_domain}' missing for idx: "
            + ", ".join(sorted(missing_center))
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=1.2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_int_list(arg: str) -> List[int]:
    values: List[int] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_groups(arg: str) -> List[List[int]]:
    groups: List[List[int]] = []
    for chunk in arg.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append([int(v) for v in chunk.replace(",", " ").split()])
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Visualize per-idx domain shifts with PCA and t-SNE embeddings."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/mvtec_ad_2/can/test_public/bad",
        help="Folder that contains <idx>_<domain>.png images.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./domain_shift_idx_plots", help="Where to save figures."
    )
    parser.add_argument(
        "--encoder_name", type=str, default="dinov2reg_vit_base_14", help="Backbone name for ViT encoder."
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
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
        help="Groups of layer indices (matching target layers order) fused together, separated by '|'.",
    )
    parser.add_argument(
        "--center_domain",
        type=str,
        default="regular",
        help="Domain name treated as the reference point per idx.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=2000)
    parser.add_argument("--tsne_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"])
    parser.add_argument("--tsne_exaggeration", type=float, default=12.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_angle", type=float, default=0.5)
    parser.add_argument(
        "--no_pca_tsne", action="store_true", help="Skip PCA reduction before feeding t-SNE."
    )
    parser.add_argument(
        "--l2norm_before_tsne",
        action="store_true",
        help="Row-wise L2 normalize before t-SNE (recommended for cosine metric).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run feature extraction on.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1)
    device = torch.device(args.device)

    target_layers = parse_int_list(args.target_layers)
    fuse_groups = parse_groups(args.fuse_groups)
    if not target_layers:
        raise ValueError("target_layers cannot be empty.")
    if not fuse_groups:
        raise ValueError("fuse_groups cannot be empty.")

    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)
    dataset = IndexedDomainDataset(args.image_dir, data_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_encoder_model(args.encoder_name, target_layers, fuse_groups, device)
    features, meta = extract_features_with_meta(model, loader, device)

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
