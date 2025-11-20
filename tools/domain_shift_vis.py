import os
import glob
import argparse
import math
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path for absolute imports when running from tools/
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import get_data_transforms
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from functools import partial


def set_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".JPG", ".JPEG"}


def list_images(root: str) -> List[str]:
    files = []
    for ext in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    files = sorted(list({os.path.abspath(p) for p in files}))
    return files


class FlatImageFolder(Dataset):
    def __init__(self, root: str, transform=None):
        self.paths = list_images(root)
        self.transform = transform
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, p


def build_encoder_model(encoder_name: str,
                        target_layers: List[int],
                        fuse_layer_encoder: List[List[int]],
                        device: torch.device) -> ViTill:
    """Construct ViTill with encoder only (no-op bottleneck/decoder)."""
    encoder = vit_encoder.load(encoder_name)

    # Determine embed dim / heads from name (align with training scripts)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("encoder_name must include one of: small/base/large")

    # Use empty bottleneck & decoder; skip decoder fusion
    bottleneck = nn.ModuleList([])
    decoder = nn.ModuleList([])
    fuse_layer_decoder: List[List[int]] = []

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        mask_neighbor_size=0,
        remove_class_token=False,
    ).to(device)
    model.eval()
    return model


def extract_features(model: ViTill, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[str]]:
    """Extract fused encoder features (patch-only), per image returns a vector.
    - ViTill.forward() returns en (list of BxCxHxW). We GAP each (HxW) and concat across groups.
    """
    feats: List[torch.Tensor] = []
    paths: List[str] = []
    with torch.no_grad():
        for imgs, batch_paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            en, _ = model(imgs)
            pooled = [f.mean(dim=[2, 3]) for f in en]  # each: BxC
            vec = torch.cat(pooled, dim=1).cpu()        # Bx(sum C)
            feats.append(vec)
            paths.extend(batch_paths)
    X = torch.cat(feats, dim=0).numpy()
    return X, paths


def pairwise_fid(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Compute Fréchet distance between two feature sets."""
    from scipy import linalg
    mu1, mu2 = Xa.mean(axis=0), Xb.mean(axis=0)
    sigma1 = np.cov(Xa, rowvar=False)
    sigma2 = np.cov(Xb, rowvar=False)
    diff = mu1 - mu2
    # product may be singular; add tiny eps for stability
    eps = 1e-6
    covmean, _ = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])) @
                              (sigma2 + eps * np.eye(sigma2.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    XX = np.sum(X * X, axis=1)[:, None]
    YY = np.sum(Y * Y, axis=1)[None, :]
    distances = XX + YY - 2 * (X @ Y.T)
    return np.exp(-gamma * distances)


def median_heuristic_gamma(X: np.ndarray) -> float:
    # Sample subset to avoid O(N^2) on large sets
    idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    Xs = X[idx]
    d2 = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0])
    if not np.isfinite(med) or med <= 0:
        med = np.mean(d2)
    gamma = 1.0 / (2.0 * max(med, 1e-6))
    return gamma


def mmd_rbf(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Biased MMD^2 with RBF kernel and median heuristic."""
    gamma = median_heuristic_gamma(np.vstack([Xa, Xb]))
    Kxx = _rbf_kernel(Xa, Xa, gamma)
    Kyy = _rbf_kernel(Xb, Xb, gamma)
    Kxy = _rbf_kernel(Xa, Xb, gamma)
    n, m = len(Xa), len(Xb)
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1) + 1e-9) \
         + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1) + 1e-9) \
         - 2 * Kxy.mean()
    return float(max(mmd2, 0.0))


def linear_auc_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Cross-validated AUC of a linear logistic regression."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aucs: List[float] = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        Xtrn = scaler.fit_transform(Xtr)
        Xten = scaler.transform(Xte)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Xtrn, ytr)
        prob = clf.predict_proba(Xten)[:, 1]
        auc = roc_auc_score(yte, prob)
        aucs.append(auc)
    return float(np.mean(aucs))


def plot_scatter(Z: np.ndarray, domain: np.ndarray, anom: np.ndarray, out_path: str, title: str):
    colors = np.array([[57/255, 106/255, 177/255],  # train-normal: blue
                       [62/255, 150/255, 81/255],   # test-normal: green
                       [204/255, 37/255, 41/255]])  # test-abnormal: red
    labels = np.array(["train-normal", "test-normal", "test-abnormal"])

    plt.figure(figsize=(7, 6))
    for d in [0, 1, 2]:
        idx = np.where(domain == d)[0]
        if len(idx) == 0:
            continue
        plt.scatter(Z[idx, 0], Z[idx, 1], s=10, c=[colors[d]], label=labels[d], alpha=0.7, edgecolors='none')
    plt.title(title)
    plt.legend(markerscale=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Dinomaly domain shift visualization and metrics")
    parser.add_argument("--train_normal_dir", type=str,
                        default="data/mvtec_ad_2/can/train/good")
    parser.add_argument("--test_normal_dir", type=str,
                        default="data/mvtec_ad_2/can/test_public/good")
    parser.add_argument("--test_abnormal_dir", type=str,
                        default="data/mvtec_ad_2/can/test_public/bad")
    parser.add_argument("--encoder_name", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--output_dir", type=str, default="./domain_shift_out")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=2000)
    parser.add_argument("--tsne_metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric for t-SNE.")
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"],
                        help="Initialization for t-SNE embedding.")
    parser.add_argument("--tsne_exaggeration", type=float, default=12.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_angle", type=float, default=0.5)
    parser.add_argument("--no_pca_tsne", action="store_true", help="Do not PCA-reduce before t-SNE.")
    parser.add_argument("--l2norm_before_tsne", action="store_true",
                        help="Row-wise L2 normalize features before t-SNE (recommended with cosine metric).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1)
    device = torch.device(args.device)

    # Data transforms consistent with training
    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)

    # Datasets & loaders
    ds_train = FlatImageFolder(args.train_normal_dir, transform=data_transform)
    ds_testn = FlatImageFolder(args.test_normal_dir, transform=data_transform)
    ds_testa = FlatImageFolder(args.test_abnormal_dir, transform=data_transform)

    def mk_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ld_train = mk_loader(ds_train)
    ld_testn = mk_loader(ds_testn)
    ld_testa = mk_loader(ds_testa)

    # Model consistent with training (encoder only; use multi-layer fused patch features)
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    model = build_encoder_model(args.encoder_name, target_layers, fuse_layer_encoder, device)

    # Extract features
    Xtr, _ = extract_features(model, ld_train, device)
    Xtn, _ = extract_features(model, ld_testn, device)
    Xta, _ = extract_features(model, ld_testa, device)

    # Save raw features
    np.save(os.path.join(args.output_dir, "feat_train_normal.npy"), Xtr)
    np.save(os.path.join(args.output_dir, "feat_test_normal.npy"), Xtn)
    np.save(os.path.join(args.output_dir, "feat_test_abnormal.npy"), Xta)

    # Build labels for visualization/metrics
    X = np.vstack([Xtr, Xtn, Xta])
    domain = np.concatenate([
        np.zeros(len(Xtr), dtype=int),
        np.ones(len(Xtn), dtype=int),
        np.full(len(Xta), 2, dtype=int)
    ])
    anom = np.concatenate([
        np.zeros(len(Xtr), dtype=int),
        np.zeros(len(Xtn), dtype=int),
        np.ones(len(Xta), dtype=int)
    ])

    # Standardize then PCA->TSNE for stability
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    pca2 = PCA(n_components=2, random_state=0)
    Zp = pca2.fit_transform(Xn)
    plot_scatter(Zp, domain, anom, os.path.join(args.output_dir, "pca.png"), "PCA (2D)")

    # Prepare input for t-SNE
    Xt = X.copy()
    if args.l2norm_before_tsne or args.tsne_metric == "cosine":
        # L2-normalize rows preserves cosine geometry
        norms = np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
        Xt = Xt / norms
    else:
        # For Euclidean, standardize
        Xt = Xn

    if args.no_pca_tsne:
        Xtsne_in = Xt
    else:
        pca_k = min(50, Xt.shape[1])
        pca50 = PCA(n_components=pca_k, random_state=0)
        Xtsne_in = pca50.fit_transform(Xt)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        metric=args.tsne_metric,
        n_iter=args.tsne_iter,
        init=args.tsne_init,
        learning_rate=args.tsne_lr,
        early_exaggeration=args.tsne_exaggeration,
        angle=args.tsne_angle,
        random_state=0,
    )
    Zt = tsne.fit_transform(Xtsne_in)
    plot_scatter(Zt, domain, anom, os.path.join(args.output_dir, "tsne.png"), "t-SNE (from PCA-50)")

    # Quantitative metrics
    # Train-normal vs Test-normal
    y_tr_tn = np.concatenate([np.zeros(len(Xtr), dtype=int), np.ones(len(Xtn), dtype=int)])
    X_tr_tn = np.vstack([Xtr, Xtn])
    auc_tr_tn = linear_auc_cv(X_tr_tn, y_tr_tn, n_splits=5)
    fid_tr_tn = pairwise_fid(Xtr, Xtn)
    mmd_tr_tn = mmd_rbf(Xtr, Xtn)

    # Train-normal vs Test-abnormal (for reference)
    y_tr_ta = np.concatenate([np.zeros(len(Xtr), dtype=int), np.ones(len(Xta), dtype=int)])
    X_tr_ta = np.vstack([Xtr, Xta])
    auc_tr_ta = linear_auc_cv(X_tr_ta, y_tr_ta, n_splits=5)
    fid_tr_ta = pairwise_fid(Xtr, Xta)
    mmd_tr_ta = mmd_rbf(Xtr, Xta)

    # Persist metrics
    metrics = {
        "train_normal_vs_test_normal": {
            "AUC_logistic": auc_tr_tn,
            "FID": fid_tr_tn,
            "MMD": mmd_tr_tn,
            "n_train": int(len(Xtr)),
            "n_test_normal": int(len(Xtn)),
        },
        "train_normal_vs_test_abnormal": {
            "AUC_logistic": auc_tr_ta,
            "FID": fid_tr_ta,
            "MMD": mmd_tr_ta,
            "n_train": int(len(Xtr)),
            "n_test_abnormal": int(len(Xta)),
        },
    }

    np.save(os.path.join(args.output_dir, "metrics.npy"), metrics)

    # Also write a readable txt
    lines = []
    lines.append("== Domain Shift Metrics ==\n")
    lines.append(f"train vs test-normal: AUC={auc_tr_tn:.4f}, FID={fid_tr_tn:.4f}, MMD={mmd_tr_tn:.4f}\n")
    lines.append(f"train vs test-abnormal: AUC={auc_tr_ta:.4f}, FID={fid_tr_ta:.4f}, MMD={mmd_tr_ta:.4f}\n")
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.writelines(lines)

    print("Saved:")
    print(" -", os.path.join(args.output_dir, "pca.png"))
    print(" -", os.path.join(args.output_dir, "tsne.png"))
    print(" -", os.path.join(args.output_dir, "metrics.txt"))


if __name__ == "__main__":
    main()
