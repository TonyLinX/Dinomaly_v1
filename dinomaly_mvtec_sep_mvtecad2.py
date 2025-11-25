# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import glob
import tifffile
import json

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, WarmCosineScheduler, \
    cal_anomaly_maps, get_gaussian_kernel
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

from preprocessing import attach_preprocessing

warnings.filterwarnings("ignore")


# encoder_name = 'dinov2reg_vit_small_14'
# encoder_name = 'dinov2reg_vit_base_14'
# encoder_name = 'dinov2reg_vit_large_14'
# ===================== Global experiment hyper-parameters =====================

SEED = 1

# Data / image
IMAGE_SIZE = 448
CROP_SIZE = 392  # crop_size == IMAGE_SIZE 等於「不實際裁掉內容」

# Transform options
USE_CENTER_CROP = True

# Model
DEFAULT_ENCODER_NAME = 'dinov2reg_vit_small_14'
ENCODER_NAME = DEFAULT_ENCODER_NAME
TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Training
TOTAL_ITERS = 5000
BATCH_SIZE = 16
LR = 2e-3
LR_FINAL = 2e-4
WARMUP_ITERS = 100
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-8
GRAD_CLIP_MAX_NORM = 0.1

# Loss (global_cosine_hm_percent)
P_FINAL = 0.9
LOSS_FACTOR = 0.1

# Inference / heatmap
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 4
INFER_BATCH_SIZE = 1


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(args, item_list, use_center_crop: bool):
    """
    Save the effective experiment configuration so this run is fully reproducible.
    """
    exp_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")

    effective_crop = CROP_SIZE if use_center_crop else IMAGE_SIZE

    config = {
        "data": {
            "data_path": args.data_path,
            "objects": item_list,
            "image_size": IMAGE_SIZE,
            "crop_size": effective_crop,
            "use_center_crop": use_center_crop,
        },
        "model": {
            "encoder_name": ENCODER_NAME,
            "target_layers": TARGET_LAYERS,
            "fuse_layer_encoder": FUSE_LAYER_ENCODER,
            "fuse_layer_decoder": FUSE_LAYER_DECODER,
        },
        "training": {
            "seed": SEED,
            "total_iters": TOTAL_ITERS,
            "batch_size": BATCH_SIZE,
            "optimizer": {
                "name": "StableAdamW",
                "lr": LR,
                "betas": list(BETAS),
                "weight_decay": WEIGHT_DECAY,
                "eps": ADAM_EPS,
            },
            "scheduler": {
                "name": "WarmCosineScheduler",
                "base_value": LR,
                "final_value": LR_FINAL,
                "total_iters": TOTAL_ITERS,
                "warmup_iters": WARMUP_ITERS,
            },
            "loss": {
                "name": "global_cosine_hm_percent",
                "p_final": P_FINAL,
                "factor": LOSS_FACTOR,
            },
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        },
        "inference": {
            "split": "test_public",
            "infer_batch_size": INFER_BATCH_SIZE,
            "gaussian_kernel_size": GAUSSIAN_KERNEL_SIZE,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "heatmap_normalization": "per_image_min_max_0_1",
            "output_dtype": "float16",
            "output_format": "tiff",
            "output_root": os.path.join(args.save_dir, args.save_name, "anomaly_images_test_public"),
        },
        "experiment": {
            "save_dir": args.save_dir,
            "save_name": args.save_name,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


class MVTecAD2TestPublicDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.img_paths = []

        for defect_type in ['good', 'bad']:
            img_dir = os.path.join(self.root, defect_type)
            if not os.path.isdir(img_dir):
                continue
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
                self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))

        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        img = self.transform(img)
        return img, (height, width), img_path


def train(item):
    setup_seed(SEED)
    print_fn(item)
    total_iters = TOTAL_ITERS
    batch_size = BATCH_SIZE
    image_size = IMAGE_SIZE
    crop_size = CROP_SIZE if USE_CENTER_CROP else IMAGE_SIZE

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    data_transform = attach_preprocessing(data_transform)

    train_path = os.path.join(args.data_path, item, 'train')
    # test_path = os.path.join(args.data_path, item)

    train_data = ImageFolder(root=train_path, transform=data_transform)
    # test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test_public")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12,
                                                   drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12)

    encoder_name = ENCODER_NAME
    # encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    target_layers = TARGET_LAYERS
    fuse_layer_encoder = FUSE_LAYER_ENCODER
    fuse_layer_decoder = FUSE_LAYER_DECODER
    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY, amsgrad=True, eps=ADAM_EPS)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=LR, final_value=LR_FINAL, total_iters=total_iters,
                                       warmup_iters=WARMUP_ITERS)

    print_fn('train image number:{}'.format(len(train_data)))

    auroc_sp = ap_sp = f1_sp = auroc_px = ap_px = f1_px = aupro_px = 0.0
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p = min(P_FINAL * it / 1000, P_FINAL)
            loss = global_cosine_hm_percent(en, de, p=p, factor=LOSS_FACTOR)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # if (it + 1) % 5000 == 0:
            #     results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            #     auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
            #
            #     print_fn(
            #         '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            #             item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
            #     model.train()

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    ckpt_path = os.path.join(
        args.save_dir,
        args.save_name,
        'checkpoints',
        f'{item}_model_{total_iters}.pth'
    )
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # Inference on test_public split and save anomaly maps as .tiff
    model.eval()
    gaussian_kernel = get_gaussian_kernel(kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA).to(device)
    test_root = os.path.join(args.data_path, item, 'test_public')
    test_data = MVTecAD2TestPublicDataset(root=test_root, transform=data_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=INFER_BATCH_SIZE, shuffle=False, num_workers=4,
                                                  drop_last=False)

    with torch.no_grad():
        for img, orig_size, img_path in test_dataloader:
            img = img.to(device)
            height, width = orig_size[0].item(), orig_size[1].item()

            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            anomaly_map = gaussian_kernel(anomaly_map)

            anomaly_map = F.interpolate(anomaly_map, size=(height, width), mode='bilinear', align_corners=False)
            anomaly_map = anomaly_map[0, 0].cpu().numpy().astype(np.float32)

            # 保留模型原生輸出，避免 per-image min-max 正規化破壞 ROC/AUPRO 排序
            # a_min, a_max = anomaly_map.min(), anomaly_map.max()
            # if a_max > a_min:
            #     anomaly_map = (anomaly_map - a_min) / (a_max - a_min)
            # else:
            #     anomaly_map = np.zeros_like(anomaly_map, dtype=np.float32)

            anomaly_map_f16 = anomaly_map.astype(np.float16)

            rel_path = os.path.relpath(img_path[0], args.data_path)
            rel_dir, filename = os.path.split(rel_path)
            basename, _ = os.path.splitext(filename)
            out_rel = os.path.join(rel_dir, basename + '.tiff')

            out_dir = os.path.join(args.save_dir, args.save_name, 'anomaly_images_test_public', os.path.dirname(out_rel))
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(args.save_dir, args.save_name, 'anomaly_images_test_public', out_rel)

            tifffile.imwrite(out_path, anomaly_map_f16)

    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='./data/mvtec_ad_2')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_sep_dinov2br_c392_en29_bn4dp2_de8_elaelu_md2_i1_it10k_sadm2e3_wd1e4_w1hcosa_ghmp09f01w1k_b16_ev_s1')
    parser.add_argument('--no_center_crop', action='store_true',
                        help='Disable center crop (use full resized image).')
    parser.add_argument('--encoder_name', type=str, default=DEFAULT_ENCODER_NAME,
                        help='Backbone encoder identifier (e.g., dinov2reg_vit_small_14/base_14/large_14).')
    args = parser.parse_args()

    # Override encoder and crop behaviour from CLI so rest of script uses chosen settings.
    ENCODER_NAME = args.encoder_name
    USE_CENTER_CROP = not args.no_center_crop
    if not USE_CENTER_CROP:
        print(f'[Config] Center crop disabled; using image_size={IMAGE_SIZE} as effective crop.')

    item_list = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']
    # item_list = ['leather']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # save experiment configuration once per run
    save_config(args, item_list, USE_CENTER_CROP)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = []
    for i, item in enumerate(item_list):
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = train(item)
        result_list.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px))
