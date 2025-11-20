#!/usr/bin/env python3
"""
Dinomaly (single-GPU, per-object) trainer with iteration-based training,
train/val loss tracking (validation uses normal images), optional AMP/TF32,
robust logging, seeded DataLoader workers, and best-checkpoint selection by
lowest val loss.

Key changes vs your previous script:
- Use official `MVTecAD2` directly for train/validation (no subclass).
- Logger is passed into `train()`; no global `print_fn` dependency.
- Iteration-based schedule preserved; debug mode shrinks iters.
- Train DataLoader uses shuffle/pin_memory/persistent_workers.
- Seeded DataLoader workers (reproducibility).
- Optional `--amp` (PyTorch autocast + GradScaler) and `--tf32`.
- Compute `val_loss` every `--eval-step` and keep `best.pth` by min val loss.
- `CUDA_LAUNCH_BLOCKING=1` only when `--debug`.

Run (example):
python dinomlay_mvtec2_sep_singal_gpu.py \
  --data_path /DATA2/mvtec_ad_2 \
  --save_dir /DATA2/outputs \
  --save_name exp_0901 \
  --obj can \
  --total-iters 10000 --eval-step 5000 \
  --batch-size 16 --num-workers 8 \
  --amp --tf32
"""

import os
import math
import argparse
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import warnings
from torch import amp as torch_amp

warnings.filterwarnings("ignore")

# ===== Project imports (keep as-is from your repo) =====
from MVTecAD2_public_code_utils.mvtec_ad_2_public_offline import MVTecAD2
from models.uad import ViTill  # ViTillv2 available if you want to switch later
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dinov1.utils import trunc_normal_
from src.utils import (
    WarmCosineScheduler,
    random_sliding_crop,
    global_cosine_hm_percent,
)


# ===================== Logger =====================

def get_logger(name: str, save_path: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fmt = logging.Formatter('%(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# ===================== Seeding =====================

def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # Make dataloader workers deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ===================== Data helpers =====================

def make_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return data_transform


def collate_img_list(batch):
    # batch: list of dicts from MVTecAD2, each with key 'sample'
    imgs = [item['sample'] for item in batch]
    # return list so we can random_sliding_crop per image inside the train loop
    return imgs

# ===================== Model factory =====================

def build_model(encoder_name: str, target_layers: List[int], fuse_enc: List[List[int]], fuse_dec: List[List[int]], device: torch.device) -> Tuple[nn.Module, int]:
    encoder = vit_encoder.load(encoder_name)
    for p in encoder.parameters():
        p.requires_grad_(False)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise ValueError("encoder_name must contain small/base/large")

    # bottleneck / decoder (trainable)
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = []
    for _ in range(8):
        blk = VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention2,
        )
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_enc,
        fuse_layer_decoder=fuse_dec,
    ).to(device)

    # init
    for m in nn.ModuleList([bottleneck, decoder]).modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    return model, embed_dim

# ===================== Val =====================
@torch.no_grad()
def evaluate_val_loss(model: nn.Module, val_loader: DataLoader, device: torch.device,
                      p_final: float, it: int, patch_size: int, stride: int,
                      amp_enabled: bool, amp_dtype: torch.dtype) -> float:
    model.eval()
    losses = []
    for imgs in val_loader:
        patches = [random_sliding_crop(img, patch_size, stride) for img in imgs]
        x = torch.stack(patches).to(device, non_blocking=True)
        use_amp = (amp_enabled and device.type == "cuda")
        
        with torch_amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            en, de = model(x)
            p = min(p_final * it / 1000.0, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.0)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if len(losses) else 0.0

# ===================== Train =====================

def train(args: argparse.Namespace, device: torch.device, logger: logging.Logger):
    log = logger.info

    # Seeding
    setup_seed(args.seed)

    # Data
    tf = make_transforms(args.image_size)

    train_ds = MVTecAD2(mad2_object=args.obj, split='train', data_root=args.data_path, transform=tf)
    val_ds   = MVTecAD2(mad2_object=args.obj, split='validation', data_root=args.data_path, transform=tf)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_img_list,
        drop_last=args.drop_last,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_img_list,
        drop_last=False,
    )

    log(f"train images: {len(train_ds)} | val images: {len(val_ds)}")

    # Model
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_enc = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_dec = [[0, 1, 2, 3], [4, 5, 6, 7]]

    model, _ = build_model(args.encoder, target_layers, fuse_enc, fuse_dec, device)

    #　trainable參數數量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {n_params/1e6:.2f}M")
    
        
    # Optimizer / LR Scheduler / AMP
    trainable = nn.ModuleList([model.bottleneck, model.decoder]) if hasattr(model, 'bottleneck') else model
    optimizer = torch.optim.AdamW([{'params': trainable.parameters()}], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True, eps=1e-10)
    lr_sched = WarmCosineScheduler(optimizer, base_value=args.lr, final_value=args.lr_final, total_iters=args.total_iters, warmup_iters=args.warmup)
    use_amp = (args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    use_scaler = (use_amp and args.amp_dtype == 'fp16')  
    scaler = torch_amp.GradScaler(enabled=use_scaler)

    log(f"device: {device} | amp: {args.amp} ({amp_dtype}) | tf32: {args.tf32}")
    
    # Iters config
    total_iters = args.total_iters if not args.debug else min(args.total_iters, 50)
    eval_step   = args.eval_step if not args.debug else min(args.eval_step, 20)

    # Training loop
    best_val = math.inf
    it = 0
    epoch = 0

    model.train()
    while it < total_iters:
        # log(f"[epoch {epoch}] start, it={it}")
        loss_buf: List[float] = []

        for imgs in train_loader:
            # batch prep: random sliding crop per image, then stack
            patches = [random_sliding_crop(img, args.image_size, args.stride) for img in imgs]
            x = torch.stack(patches).to(device, non_blocking=True)

            # schedule p
            p = min(args.p_final * it / 1000.0, args.p_final)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch_amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    en, de = model(x)
                    loss = global_cosine_hm_percent(en, de, p=p, factor=args.hm_factor)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 先反縮放
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                en, de = model(x)
                loss = global_cosine_hm_percent(en, de, p=p, factor=args.hm_factor)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            lr_sched.step()

            loss_buf.append(float(loss.detach().cpu()))
            it += 1

            if it % args.log_step == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                log(f"iter {it:6d}/{total_iters} | lr {cur_lr:.6e} | train_loss {np.mean(loss_buf):.6f}")

            if it % eval_step == 0:
                # checkpoint (iter)
                ckpt_dir = Path(args.save_dir) / args.save_name / args.obj
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"iter_{it:06d}.pth")

                # validation
                val_loss = evaluate_val_loss(
                    model, val_loader, device,
                    args.p_final, it, args.image_size, args.stride,
                    amp_enabled=args.amp, amp_dtype=amp_dtype
                )
                log(f"[val] iter {it:6d} | val_loss {val_loss:.6f}")
                model.train()
                
                # best model by min val loss
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), ckpt_dir / "best.pth")
                    log(f"[ckpt] best.pth updated (val_loss={best_val:.6f})")

            if it >= total_iters:
                break

        # epoch end
        log(f"[epoch {epoch}] it={it}/{total_iters} | train_loss_epoch_avg={np.mean(loss_buf):.6f}")
        epoch += 1

    # final save
    ckpt_dir = Path(args.save_dir) / args.save_name / args.obj
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "last.pth")
    log("[done] saved last.pth")

# ===================== CLI =====================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Dinomaly iteration-based trainer (train/val loss)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./saved_results')
    ap.add_argument('--save_name', type=str, default='experiment')
    ap.add_argument('--obj', type=str, required=True, help='Object class (e.g., can, fabric, ...)')

    # Training sizes
    ap.add_argument('--image_size', type=int, default=448)
    ap.add_argument('--stride', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--drop-last', action='store_true', help='Drop last incomplete batch for training')

    # Optim & schedule
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr_final', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--grad_clip', type=float, default=0.1)

    # Hard-mining & loss
    ap.add_argument('--p_final', type=float, default=0.9)
    ap.add_argument('--hm_factor', type=float, default=0.1, help='gradient scaling factor for hard-mining (0.0 disables hook)')

    # Iteration control
    ap.add_argument('--total-iters', type=int, default=10000)
    ap.add_argument('--eval-step', type=int, default=2500)
    ap.add_argument('--log-step', type=int, default=100)

    # Model
    ap.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14', help='dinov2reg_vit_small_14/base_14/large_14')

    # System
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--amp-dtype', type=str, default='bf16', choices=['fp16','bf16'])
    ap.add_argument('--tf32', action='store_true')
    ap.add_argument('--debug', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

    # Debug-only CUDA sync to ease crash/NaN tracing
    if args.debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Device & TF32
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')

    # Logger per object exp dir
    log_dir = os.path.join(args.save_dir, args.save_name, args.obj)
    logger = get_logger(args.save_name, log_dir)

    # Start
    train(args, device, logger)


if __name__ == '__main__':
    main()
