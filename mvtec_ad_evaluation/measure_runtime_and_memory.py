"""
Script to measure inference runtime [seconds] and GPU memory consumption [MiB].
Note that this script does not support measuring memory consumption when 
deploying models on CPU.

general usage:
- specify OUT_DIR
- insert your model and forward pass call (see # INSERT)
- run this script
"""

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import time
import argparse
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
from models.Necker import Necker
from models.Adapter import Adapter
from models.MapMaker import MapMaker
from models.CoOp import PromptMaker
from easydict import EasyDict
from utils.misc_helper import map_func, set_seed
import open_clip

USE_GPU = True
BATCH_SIZE = 1
WARMUP_ITERATIONS_GPU = 1000
TIMING_ITERATIONS_GPU = 1000
TIMING_ITERATIONS_CPU = 1000  # no warmup for CPU
OUT_DIR = 'metrics'  # INSERT

@torch.no_grad()
def make_vision_tokens_info(model, model_cfg, layers_out):
    img = torch.ones(
        (1, 3, model_cfg["vision_cfg"]["image_size"], model_cfg["vision_cfg"]["image_size"])
    ).to(model.device)
    _, tokens = model.encode_image(img, layers_out)
    if len(tokens[0].shape) == 3:
        model.token_size = [int(math.sqrt(t.shape[1] - 1)) for t in tokens]
        model.token_c = [t.shape[-1] for t in tokens]
    else:
        model.token_size = [t.shape[2] for t in tokens]
        model.token_c = [t.shape[1] for t in tokens]
    model.embed_dim = model_cfg["embed_dim"]
    

class InfiniteDataset(torch.utils.data.IterableDataset):

    def __init__(self, image_height=256, image_width=256, dtype=torch.float32):
        self.dtype = dtype
        self.image_height = image_height
        self.image_width = image_width
        super().__init__()

    def __iter__(self):
        while True:
            image_np = np.random.randn(3, self.image_height, self.image_width)
            image_pt = torch.as_tensor(image_np, dtype=self.dtype)
            yield image_pt


def main(args):
    with open(args.config_path) as f:
        args.config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    set_seed(seed=args.config.get("random_seed", 0))
    device = torch.device('cpu')
    device_name = 'cpu'
    if USE_GPU:
        assert torch.cuda.is_available(), 'GPU not available.'
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        device_name = 'gpu'

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Using device {device}')

    torch.set_grad_enabled(False)

    measurement_info = {
        'image_height': [],
        'image_width': [],
        'runtime_mean': [],
        'runtime_std': [],
        'runtime_min': [],
        'runtime_max': [],
        'peak_memory': [],
    }

    num_iterations = (
        WARMUP_ITERATIONS_GPU + TIMING_ITERATIONS_GPU
        if USE_GPU
        else TIMING_ITERATIONS_CPU
    )

    for image_size in [(224,224)]:

        img_height, img_width = image_size
        measurement_info['image_height'].append(img_height)
        measurement_info['image_width'].append(img_width)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # INSERT
        # create your model instance here
        model, preprocess, model_cfg = open_clip.create_model_and_transforms(
            args.config.model_name, args.config.image_size, device=device
        )
        make_vision_tokens_info(model, model_cfg, args.config.layers_out)
        necker = Necker(clip_model=model).to(device)
        adapter = Adapter(clip_model=model, target=model_cfg["embed_dim"]).to(device)
        prompt_maker = PromptMaker(
            prompts=args.config.prompts,
            clip_model=model,
            n_ctx=args.config.n_learnable_token,
            CSC=args.config.CSC,
            class_token_position=args.config.class_token_positions,
        ).to(device)
        map_maker = MapMaker(image_size=args.config.image_size).to(device)
    
        ckpt = torch.load(args.checkpoint_path, map_location=map_func)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        prompt_maker.prompt_learner.load_state_dict(ckpt["prompt_state_dict"])
        prompt_maker.prompt_learner.eval()
        adapter.eval()

        dataset = InfiniteDataset(
            image_height=img_height,
            image_width=img_width,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True
        )

        dataiter = iter(dataloader)
        times = []

        timing_successful = True
        print(f'\nInference on image size (hxw): {img_height}x{img_width}')
        try:
            for _ in tqdm(range(num_iterations)):
                input_tensor = next(dataiter)

                # START
                start_time = time.time()

                input_tensor = input_tensor.to(device)

                # INSERT
                # call your forward pass and ensure
                # that in the end you have an anomaly image on the cpu
                _, image_tokens = model.encode_image(input_tensor, out_layers=args.config.layers_out)
                image_features = necker(image_tokens)
                vision_adapter_features = adapter(image_features)
                prompt_adapter_features = prompt_maker(vision_adapter_features)
                anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
                anomaly_image = anomaly_map[:, 1, :, :].cpu()

                # END
                # we stop the timing as soon as we have the result on the cpu
                end_time = time.time()
                times.append((end_time - start_time) / BATCH_SIZE)
        except Exception as error:
            timing_successful = False
            print('timing not successful:', error)

            error_value = np.nan
            measurement_info['runtime_mean'].append(error_value)
            measurement_info['runtime_std'].append(error_value)
            measurement_info['runtime_min'].append(error_value)
            measurement_info['runtime_max'].append(error_value)
            measurement_info['peak_memory'].append(error_value)

        if timing_successful:

            if USE_GPU:
                # get peak memory
                peak_memory = torch.cuda.memory_stats()[
                    'reserved_bytes.all.peak'
                ] / (1024**2)
                measurement_info['peak_memory'].append(peak_memory)
                print('Peak Memory:', peak_memory)

                # discard warmup iterations
                times = times[WARMUP_ITERATIONS_GPU:]

            else:
                measurement_info['peak_memory'].append(np.nan)

            print('Mean runtime [s]:', np.mean(times))
            print('Std:', np.std(times))
            print('Min:', np.min(times))
            print('Max:', np.max(times))
            print('Last:', times[-1])

            measurement_info['runtime_mean'].append(np.mean(times))
            measurement_info['runtime_std'].append(np.std(times))
            measurement_info['runtime_min'].append(np.min(times))
            measurement_info['runtime_max'].append(np.max(times))

    # save the data as csv
    measurement_data = pd.DataFrame.from_dict(
        measurement_info, orient='columns'
    )
    measurement_data.to_csv(
        os.path.join(OUT_DIR, f'runtimes_and_memory_{device_name}.csv'),
        sep=',',
        index=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test MediCLIP on MVTec AD 2")
    parser.add_argument("--config_path", type=str, help="model config path")
    parser.add_argument("--checkpoint_path", type=str, help="trained checkpoint path")
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
