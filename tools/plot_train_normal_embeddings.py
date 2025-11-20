import os
import glob
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root on path for absolute imports when running from tools/
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import get_data_transforms
from models import vit_encoder
from models.uad import ViTill


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


def build_vitill(encoder_name: str,
                 target_layers: List[int],
                 fuse_layer_encoder: List[List[int]],
                 device: torch.device) -> ViTill:
    encoder = vit_encoder.load(encoder_name)
    # Empty bottleneck/decoder; only encoder features
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


@torch.no_grad()
def extract_encoder_fused_vec(model: ViTill, loader: DataLoader, device: torch.device) -> np.ndarray:
    vecs: List[torch.Tensor] = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        en, _ = model(imgs)
        pooled = [f.mean(dim=[2, 3]) for f in en]  # each BxC
        v = torch.cat(pooled, dim=1).cpu()
        vecs.append(v)
    X = torch.cat(vecs, dim=0).numpy()
    return X


@torch.no_grad()
def extract_encoder_cls_vec(encoder, loader: DataLoader, device: torch.device,
                            target_layers: List[int], fuse_layer_encoder: List[List[int]]) -> np.ndarray:
    """Extract CLS token features from specified encoder layers.
    - For each target layer, take x[:,0,:] (CLS), then fuse per fuse_layer_encoder by mean, then concat groups.
    Returns: N x (groups * C)
    """
    encoder.eval()
    vecs: List[torch.Tensor] = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        x = encoder.prepare_tokens(imgs)
        cls_list = []  # list of BxC
        for i, blk in enumerate(encoder.blocks):
            x = blk(x)
            if i in target_layers:
                cls_vec = x[:, 0, :]  # BxC
                cls_list.append(cls_vec)
            if i >= max(target_layers):
                break
        # fuse per group
        group_feats: List[torch.Tensor] = []
        for idxs in fuse_layer_encoder:
            # pick from cls_list by relative indices
            selected = [cls_list[j] for j in idxs]
            g = torch.stack(selected, dim=1).mean(dim=1)  # BxC
            group_feats.append(g)
        v = torch.cat(group_feats, dim=1).cpu()  # Bx(groups*C)
        vecs.append(v)
    X = torch.cat(vecs, dim=0).numpy()
    return X


def make_scatter(Z: np.ndarray, title: str, out_path: str, color=(57/255, 106/255, 177/255)):
    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=8, c=[color], alpha=0.8, edgecolors='none')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_side_by_side(Zp: np.ndarray, Zt: np.ndarray, title_left: str, title_right: str, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(Zp[:, 0], Zp[:, 1], s=8, c=[[57/255, 106/255, 177/255]], alpha=0.8, edgecolors='none')
    axes[0].set_title(title_left)
    axes[1].scatter(Zt[:, 0], Zt[:, 1], s=8, c=[[57/255, 106/255, 177/255]], alpha=0.8, edgecolors='none')
    axes[1].set_title(title_right)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot PCA/t-SNE for train-normal using Dinomaly features")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to train-normal images directory")
    parser.add_argument("--output_dir", type=str, default="./train_normal_embed_out")
    parser.add_argument("--encoder_name", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    # t-SNE params
    parser.add_argument("--tsne_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=2000)
    parser.add_argument("--tsne_init", type=str, default="pca", choices=["pca", "random"])
    parser.add_argument("--tsne_exaggeration", type=float, default=12.0)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--tsne_angle", type=float, default=0.5)
    parser.add_argument("--no_pca_tsne", action="store_true")
    parser.add_argument("--l2norm_before_tsne", action="store_true")
    parser.add_argument("--use_cls", action="store_true", help="Use CLS tokens instead of patch features")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1)

    # Data
    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)
    ds = FlatImageFolder(args.data_dir, transform=data_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 1) Fused middle layers vs 2) Last layer
    target_layers_fused = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder_fused = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Determine last index
    tmp_enc = vit_encoder.load(args.encoder_name)
    last_idx = len(tmp_enc.blocks) - 1
    del tmp_enc
    target_layers_last = [last_idx]
    fuse_layer_encoder_last = [[0]]

    if args.use_cls:
        enc = vit_encoder.load(args.encoder_name).to(device)
        X_fused = extract_encoder_cls_vec(enc, loader, device, target_layers_fused, fuse_layer_encoder_fused)
        X_last = extract_encoder_cls_vec(enc, loader, device, target_layers_last, fuse_layer_encoder_last)
        mode_tag = "cls"
    else:
        model_fused = build_vitill(args.encoder_name, target_layers_fused, fuse_layer_encoder_fused, device)
        X_fused = extract_encoder_fused_vec(model_fused, loader, device)
        model_last = build_vitill(args.encoder_name, target_layers_last, fuse_layer_encoder_last, device)
        X_last = extract_encoder_fused_vec(model_last, loader, device)
        mode_tag = "patch"

    # Save raw features
    np.save(os.path.join(args.output_dir, f"feat_train_fused_{mode_tag}.npy"), X_fused)
    np.save(os.path.join(args.output_dir, f"feat_train_last_{mode_tag}.npy"), X_last)

    def project_and_plot(X: np.ndarray, tag: str):
        # Standardize for PCA and Euclidean t-SNE; do L2 for cosine t-SNE
        scaler = StandardScaler()
        Xn = scaler.fit_transform(X)

        # PCA 2D
        pca2 = PCA(n_components=2, random_state=0)
        Zp = pca2.fit_transform(Xn)
        make_scatter(Zp, f"PCA (2D) - {tag}", os.path.join(args.output_dir, f"{tag}_pca.png"))

        # Prepare t-SNE input
        Xt = X.copy()
        if args.l2norm_before_tsne or args.tsne_metric == "cosine":
            norms = np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-12
            Xt = Xt / norms
        else:
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

        make_scatter(Zt, f"t-SNE (2D) - {tag}", os.path.join(args.output_dir, f"{tag}_tsne.png"))
        make_side_by_side(Zp, Zt, f"PCA - {tag}", f"t-SNE - {tag}", os.path.join(args.output_dir, f"{tag}_both.png"))

    project_and_plot(X_fused, "fused")
    project_and_plot(X_last, "last")

    print("Saved embeddings and plots to:", args.output_dir)


if __name__ == "__main__":
    main()
