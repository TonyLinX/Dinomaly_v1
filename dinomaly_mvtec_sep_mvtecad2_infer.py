import argparse
import glob
import logging
import os
import random
from functools import partial

import numpy as np
import tifffile
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F

from dataset import get_data_transforms, sliding_window_grid
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel

DEFAULT_ITEMS = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']

# Experiment defaults (mirrors dinomaly_mvtec_sep_mvtec_ad2.py)
SEED = 1
IMAGE_SIZE = 448
CROP_SIZE = 392  # crop_size == IMAGE_SIZE 等於「不實際裁掉內容」
USE_CENTER_CROP = True
DEFAULT_ENCODER_NAME = 'dinov2reg_vit_small_14'
ENCODER_NAME = DEFAULT_ENCODER_NAME
TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 4
INFER_BATCH_SIZE = 1


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, save_path: str | None = None, level: str = 'INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        logger.addHandler(stream_handler)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)

    return logger


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
        if self.transform is not None:
            img = self.transform(img)
        return img, (height, width), img_path


def _to_int_list(value):
    if torch.is_tensor(value):
        return [int(v) for v in value.view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _normalize_orig_size(orig_size):
    if torch.is_tensor(orig_size):
        values = orig_size.tolist()
        if isinstance(values[0], list):
            return [(int(h), int(w)) for h, w in values]
        if len(values) == 2:
            return [(int(values[0]), int(values[1]))]
    elif isinstance(orig_size, (list, tuple)):
        if len(orig_size) == 2 and all(isinstance(v, (list, tuple)) or torch.is_tensor(v) for v in orig_size):
            heights = _to_int_list(orig_size[0])
            widths = _to_int_list(orig_size[1])
            return list(zip(heights, widths))
        return [(int(h), int(w)) for h, w in orig_size]
    raise TypeError(f'Unsupported orig_size type: {type(orig_size)}')


def sliding_window_anomaly_map(model, image, gaussian_kernel, data_transform, window_size, overlap, device,
                               batch_size=1):
    width, height = image.size
    coords = sliding_window_grid(width, height, window_size, overlap)

    sum_map = torch.zeros((1, 1, height, width), device=device)
    count_map = torch.zeros((1, 1, height, width), device=device)
    patches, locs = [], []

    def flush():
        if not patches:
            return
        batch = torch.stack(patches, dim=0).to(device)
        en, de = model(batch)
        anomaly_batch, _ = cal_anomaly_maps(en, de, batch.shape[-1])
        anomaly_batch = gaussian_kernel(anomaly_batch)
        for idx, (x, y) in enumerate(locs):
            sum_map[:, :, y:y + window_size, x:x + window_size] += anomaly_batch[idx:idx + 1]
            count_map[:, :, y:y + window_size, x:x + window_size] += 1
        patches.clear()
        locs.clear()

    for x, y in coords:
        patch = image.crop((x, y, x + window_size, y + window_size))
        patch_tensor = data_transform(patch)
        patches.append(patch_tensor)
        locs.append((x, y))
        if len(patches) >= batch_size:
            flush()
    flush()

    anomaly_map = sum_map / count_map.clamp(min=1e-6)
    return anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32)


def build_model():
    encoder = vit_encoder.load(ENCODER_NAME)
    target_layers = TARGET_LAYERS

    if 'small' in ENCODER_NAME:
        embed_dim, num_heads = 384, 6
    elif 'base' in ENCODER_NAME:
        embed_dim, num_heads = 768, 12
    elif 'large' in ENCODER_NAME:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Unsupported encoder size in ENCODER_NAME.")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
    decoder = nn.ModuleList([
        VitBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn_drop=0.,
            attn=LinearAttention2
        ) for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=FUSE_LAYER_ENCODER,
        fuse_layer_decoder=FUSE_LAYER_DECODER
    )
    return model


def inference_one_item(item, args, device, data_transform):
    ckpt_name = args.checkpoint_format.format(item=item)
    ckpt_path = os.path.join(args.save_dir, args.save_name, 'checkpoints', ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found for {item}: {ckpt_path}')

    model = build_model().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    gaussian_kernel = get_gaussian_kernel(kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA).to(device)
    test_root = os.path.join(args.data_path, item, 'test_public')
    is_slide_mode = args.input_mode == 'slide'
    if is_slide_mode:
        test_data = MVTecAD2TestPublicDataset(root=test_root, transform=None)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=lambda batch: batch
        )
    else:
        test_data = MVTecAD2TestPublicDataset(root=test_root, transform=data_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=INFER_BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )

    output_root = os.path.join(args.save_dir, args.save_name, 'anomaly_images_test_public')
    os.makedirs(output_root, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            if is_slide_mode:
                img, orig_size, img_path = batch[0]
                anomaly_map = sliding_window_anomaly_map(
                    model=model,
                    image=img,
                    gaussian_kernel=gaussian_kernel,
                    data_transform=data_transform,
                    window_size=args.slide_window_size,
                    overlap=args.slide_window_overlap,
                    device=device,
                    batch_size=INFER_BATCH_SIZE
                )
                path_list = [img_path]
                anomaly_maps = [anomaly_map]
            else:
                img, orig_size, img_path = batch
                img = img.to(device)
                size_pairs = _normalize_orig_size(orig_size)
                if isinstance(img_path, (list, tuple)):
                    path_list = list(img_path)
                else:
                    path_list = [img_path]

                en, de = model(img)
                anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
                anomaly_map = gaussian_kernel(anomaly_map)

                anomaly_maps = []
                for idx, (height, width) in enumerate(size_pairs):
                    resized_map = F.interpolate(anomaly_map[idx:idx + 1], size=(height, width),
                                                mode='bilinear', align_corners=False)
                    resized_map = resized_map[0, 0].cpu().numpy().astype(np.float32)
                    anomaly_maps.append(resized_map)

            for idx, map_np in enumerate(anomaly_maps):
                anomaly_map_f16 = map_np.astype(np.float16)

                rel_path = os.path.relpath(path_list[idx], args.data_path)
                rel_dir, filename = os.path.split(rel_path)
                basename, _ = os.path.splitext(filename)
                out_rel = os.path.join(rel_dir, basename + '.tiff')

                out_dir = os.path.join(output_root, os.path.dirname(out_rel))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(output_root, out_rel)
                tifffile.imwrite(out_path, anomaly_map_f16)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference-only pipeline for MVTec AD2 Dinomaly variant.')
    parser.add_argument('--data_path', type=str, default='./data/mvtec_ad_2')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_sep_dinov2br_c392_en29_bn4dp2_de8_elaelu_md2_i1_it10k_sadm2e3_wd1e4_w1hcosa_ghmp09f01w1k_b16_ev_s1')
    parser.add_argument('--items', nargs='+', default=DEFAULT_ITEMS,
                        help='Object categories to infer. Default is all AD2 objects.')
    parser.add_argument('--checkpoint_format', type=str, default='{item}_model_5000.pth',
                        help='Filename format inside checkpoints directory. Must contain {item}.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device string (e.g., cuda:0). Defaults to CUDA if available.')
    parser.add_argument('--no_center_crop', action='store_true',
                        help='Disable center crop (use full resized image).')
    parser.add_argument('--input_mode', choices=['resize', 'slide'], default='resize',
                        help='resize: keep resize+crop; slide: tile the original image into patches.')
    parser.add_argument('--slide_window_size', type=int, default=IMAGE_SIZE,
                        help='Patch size for sliding-window mode (e.g., 448).')
    parser.add_argument('--slide_window_overlap', type=float, default=0.2,
                        help='Overlap ratio [0,1) between sliding windows.')
    parser.add_argument('--encoder_name', type=str, default=None,
                        help='Backbone encoder identifier (e.g., dinov2reg_vit_small_14/base_14/large_14). '
                             'Defaults to the encoder used during training if known, otherwise small.')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(SEED)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    global USE_CENTER_CROP, ENCODER_NAME
    USE_CENTER_CROP = (not args.no_center_crop) and args.input_mode == 'resize'

    if args.encoder_name is not None:
        ENCODER_NAME = args.encoder_name

    # center crop behaviour follows training by default, but can be changed via config edit later if needed
    effective_crop = CROP_SIZE if USE_CENTER_CROP else IMAGE_SIZE

    logger = get_logger(f'{args.save_name}_infer', os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    data_transform, _ = get_data_transforms(
        IMAGE_SIZE,
        effective_crop,
        mode='slide' if args.input_mode == 'slide' else 'resize'
    )

    print_fn(f'Running inference on device: {device}')
    if args.input_mode == 'slide':
        print_fn(f'Input mode: slide-window | window={args.slide_window_size}, overlap={args.slide_window_overlap}')
    else:
        print_fn(f'Input mode: resize | center_crop: {USE_CENTER_CROP} (image_size={IMAGE_SIZE}, crop_size={effective_crop})')
    print_fn(f'Encoder: {ENCODER_NAME}')
    for item in args.items:
        print_fn(f'=== Inference: {item} ===')
        inference_one_item(item, args, device, data_transform)

    config_path = os.path.join(args.save_dir, args.save_name, 'config.json')
    if os.path.exists(config_path):
        print_fn(f'Config reference: {config_path}')


if __name__ == '__main__':
    main()
