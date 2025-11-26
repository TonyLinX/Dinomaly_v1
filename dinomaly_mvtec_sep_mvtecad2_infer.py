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

from dataset import get_data_transforms
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


class LightInvariantProjector(nn.Module):
    """
    g(F): illumination-invariant projector
    input : [B, C, H, W]
    output: [B, C, H, W]
    """
    def __init__(self, in_channels: int, hidden_ratio: float = 2.0):
        super().__init__()
        hidden = int(in_channels * hidden_ratio)

        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(1, hidden)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(1, in_channels)

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        residual = F_in
        x = self.conv1(F_in)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        out = self.act(x + residual)
        return out


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
    projector_front = LightInvariantProjector(embed_dim)
    projector_back = LightInvariantProjector(embed_dim)
    return model, projector_front, projector_back


# ========= 共用的小工具（debug 用） =========

def load_ckpt_and_model(item, args, device):
    ckpt_name = args.checkpoint_format.format(item=item)
    ckpt_path = os.path.join(args.save_dir, args.save_name, 'checkpoints', ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found for {item}: {ckpt_path}')

    model, projector_front, projector_back = build_model()
    model = model.to(device)
    projector_front = projector_front.to(device)
    projector_back = projector_back.to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        model.load_state_dict(state_dict['model'], strict=True)
        if 'projector_front' in state_dict and 'projector_back' in state_dict:
            projector_front.load_state_dict(state_dict['projector_front'], strict=True)
            projector_back.load_state_dict(state_dict['projector_back'], strict=True)
        else:
            logging.warning(
                f'Checkpoint {ckpt_path} has no projector weights; using randomly initialized projectors.'
            )
    else:
        # baseline 沒 projector 的話，這個 script 就沒啥 debug 意義
        logging.warning(
            f'Checkpoint {ckpt_path} not in expected dict format with model/projector; loading as plain model.'
        )
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    projector_front.eval()
    projector_back.eval()
    return model, projector_front, projector_back


def load_image_tensor(path, transform, device):
    img = Image.open(path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img


@torch.no_grad()
def forward_features(model, projector_front, projector_back, img_tensor):
    """
    img_tensor: [1, 3, H, W]
    回傳: F_front, F_back, Z_front, Z_back
    """
    en, _ = model(img_tensor)
    F_front, F_back = en
    Z_front = projector_front(F_front)
    Z_back = projector_back(F_back)
    return F_front, F_back, Z_front, Z_back


def feature_rmse(F_, Z_):
    return (F_ - Z_).pow(2).mean().sqrt().item()


def feature_cosine(F1, F2):
    # B,C,H,W -> B,D
    a = F1.reshape(F1.shape[0], -1)
    b = F2.reshape(F2.shape[0], -1)
    return F.cosine_similarity(a, b).mean().item()


# ========= 原本的 anomaly map inference =========

def inference_one_item(item, args, device, data_transform):
    model, projector_front, projector_back = load_ckpt_and_model(item, args, device)

    gaussian_kernel = get_gaussian_kernel(kernel_size=GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA).to(device)
    test_root = os.path.join(args.data_path, item, 'test_public')
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
        for img, orig_size, img_path in test_loader:
            img = img.to(device)
            size_pairs = _normalize_orig_size(orig_size)
            if isinstance(img_path, (list, tuple)):
                path_list = list(img_path)
            else:
                path_list = [img_path]

            en, de = model(img)
            z = [projector_front(en[0]), projector_back(en[1])]
            anomaly_map, _ = cal_anomaly_maps(z, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for idx, (height, width) in enumerate(size_pairs):
                resized_map = F.interpolate(anomaly_map[idx:idx + 1], size=(height, width),
                                            mode='bilinear', align_corners=False)
                resized_map = resized_map[0, 0].cpu().numpy().astype(np.float32)
                anomaly_map_f16 = resized_map.astype(np.float16)

                rel_path = os.path.relpath(path_list[idx], args.data_path)
                rel_dir, filename = os.path.split(rel_path)
                basename, _ = os.path.splitext(filename)
                out_rel = os.path.join(rel_dir, basename + '.tiff')

                out_dir = os.path.join(output_root, os.path.dirname(out_rel))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(output_root, out_rel)
                tifffile.imwrite(out_path, anomaly_map_f16)


# ========= 新增的 debug mode =========

def debug_features_one_item(item, args, device, data_transform, print_fn):
    """
    (1) 看 Z 跟 F 差多遠 (RMSE)
    (2) 看 regular vs 各種 lighting 的 cosine, 在 F / Z 空間分別是多少
    """
    model, projector_front, projector_back = load_ckpt_and_model(item, args, device)

    print_fn(f'=== Feature debug for item: {item} ===')

    # ------- (1) F vs Z on train/good 000/001/002_regular -------
    train_dir = os.path.join(args.data_path, item, 'train', 'good')
    train_regular_files = [
        os.path.join(train_dir, f'{idx:03d}_regular.png') for idx in [0, 1, 2]
    ]

    rmse_front_list = []
    rmse_back_list = []
    any_found = False

    for path in train_regular_files:
        if not os.path.exists(path):
            print_fn(f'[{item}] train regular not found, skip: {path}')
            continue
        any_found = True
        img_tensor = load_image_tensor(path, data_transform, device)
        F_front, F_back, Z_front, Z_back = forward_features(
            model, projector_front, projector_back, img_tensor
        )
        r_front = feature_rmse(F_front, Z_front)
        r_back = feature_rmse(F_back, Z_back)
        rmse_front_list.append(r_front)
        rmse_back_list.append(r_back)
        print_fn(f'[{item}] F-Z RMSE {os.path.basename(path)} '
                 f'front={r_front:.5f}, back={r_back:.5f}')

    if any_found:
        print_fn(f'[{item}] mean F-Z RMSE (train regular): '
                 f'front={np.mean(rmse_front_list):.5f}, '
                 f'back={np.mean(rmse_back_list):.5f}')
    else:
        print_fn(f'[{item}] No train regular images found for F-Z RMSE check.')

    # ------- (2) regular vs over/shift/under in test_public/bad -------
    # 依你給的檔名，先嘗試這幾個；不存在就略過
    variant_names = [
        '000_regular.png',
        '000_overexposed.png',
        '000_shift_1.png',
        '000_shift_2.png',
        '000_shift_3.png',
        '000_underexposed.png',
    ]
    test_bad_dir = os.path.join(args.data_path, item, 'test_public', 'bad')

    variant_paths = {}
    for name in variant_names:
        path = os.path.join(test_bad_dir, name)
        if os.path.exists(path):
            variant_paths[name] = path

    if '000_regular.png' not in variant_paths:
        print_fn(f'[{item}] 000_regular.png not found in test_public/bad, '
                 f'skip domain-shift cosine check.')
        return

    features = {}
    for name, path in variant_paths.items():
        img_tensor = load_image_tensor(path, data_transform, device)
        F_front, F_back, Z_front, Z_back = forward_features(
            model, projector_front, projector_back, img_tensor
        )
        features[name] = {
            'F_front': F_front,
            'F_back': F_back,
            'Z_front': Z_front,
            'Z_back': Z_back,
        }
        print_fn(f'[{item}] loaded feature for {name} ({path})')

    ref_key = '000_regular.png'
    print_fn(f'[{item}] Cosine between regular (000_regular) and other lightings:')

    for name in sorted(variant_paths.keys()):
        if name == ref_key:
            continue
        f_reg = features[ref_key]
        f_var = features[name]

        cos_F_front = feature_cosine(f_reg['F_front'], f_var['F_front'])
        cos_Z_front = feature_cosine(f_reg['Z_front'], f_var['Z_front'])
        cos_F_back = feature_cosine(f_reg['F_back'], f_var['F_back'])
        cos_Z_back = feature_cosine(f_reg['Z_back'], f_var['Z_back'])

        # 簡單把 "000_" 和 ".png" 去掉，便於閱讀
        short = name.replace('000_', '').replace('.png', '')
        print_fn(
            f'  regular vs {short:12s} | '
            f'F_front={cos_F_front:.4f}, Z_front={cos_Z_front:.4f}, '
            f'F_back={cos_F_back:.4f}, Z_back={cos_Z_back:.4f}'
        )


# ========= CLI / main =========

def parse_args():
    parser = argparse.ArgumentParser(description='Inference / debug pipeline for MVTec AD2 Dinomaly variant.')
    parser.add_argument('--data_path', type=str, default='./data/mvtec_ad_2')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mvtec_sep_dinov2br_c392_en29_bn4dp2_de8_elaelu_md2_i1_it10k_sadm2e3_wd1e4_w1hcosa_ghmp09f01w1k_b16_ev_s1')
    parser.add_argument('--items', nargs='+', default=DEFAULT_ITEMS,
                        help='Object categories. Default is all AD2 objects.')
    parser.add_argument('--checkpoint_format', type=str, default='{item}_model_5000.pth',
                        help='Filename format inside checkpoints directory. Must contain {item}.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device string (e.g., cuda:0). Defaults to CUDA if available.')
    parser.add_argument('--no_center_crop', action='store_true',
                        help='Disable center crop (use full resized image).')
    parser.add_argument('--encoder_name', type=str, default=None,
                        help='Backbone encoder identifier (e.g., dinov2reg_vit_small_14/base_14/large_14). '
                             'Defaults to the encoder used during training if known, otherwise small.')
    parser.add_argument('--debug_features', action='store_true',
                        help='If set, run feature-space diagnostics (F vs Z, '
                             'regular-vs-lighting cosine) instead of writing anomaly maps.')
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(SEED)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    global USE_CENTER_CROP, ENCODER_NAME
    USE_CENTER_CROP = not args.no_center_crop

    if args.encoder_name is not None:
        ENCODER_NAME = args.encoder_name

    effective_crop = CROP_SIZE if USE_CENTER_CROP else IMAGE_SIZE

    logger = get_logger(f'{args.save_name}_infer', os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    data_transform, _ = get_data_transforms(IMAGE_SIZE, effective_crop)

    print_fn(f'Running on device: {device}')
    print_fn(f'Encoder: {ENCODER_NAME} | center_crop: {USE_CENTER_CROP} '
             f'(image_size={IMAGE_SIZE}, crop_size={effective_crop})')

    if args.debug_features:
        print_fn('*** DEBUG MODE: feature-space diagnostics ***')
        for item in args.items:
            # 你給的 path 只有 can / fabric，其它 item 找不到檔案會自動跳過 domain-shift 段
            debug_features_one_item(item, args, device, data_transform, print_fn)
    else:
        print_fn('*** INFERENCE MODE: saving anomaly maps as .tiff ***')
        for item in args.items:
            print_fn(f'=== Inference: {item} ===')
            inference_one_item(item, args, device, data_transform)

    config_path = os.path.join(args.save_dir, args.save_name, 'config.json')
    if os.path.exists(config_path):
        print_fn(f'Config reference: {config_path}')


if __name__ == '__main__':
    main()
