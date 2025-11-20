'''
MVTec AD 2 Dataset.
The dataset class provides functions to load images (and ground truth if
applicable) from the MVTec AD 2 dataset and to store anomaly images in the
correct structure for evaluating performance on the evaluation server.
'''

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

PATH_TO_MVTEC_AD_2_FOLDER = 'data/mvtec_ad_2'
MVTEC_AD_2_OBJECTS = [
    'can',
    'fabric',
    'fruit_jelly',
    'rice',
    'sheet_metal',
    'vial',
    'wallplugs',
    'walnuts',
]


class MVTecAD2(Dataset):
    """Dataset class for MVTec AD 2 objects.

    Args:
        mad2_object (str): can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
        split (str): train, validation, test_public, test_private, test_private_mixed
        transform (function, optional): transform applied to samples, defaults to 'to_tensor'
    """

    def __init__(
        self,
        mad2_object,
        split,
        data_root,
        transform=to_tensor,
    ):
        assert split in {
            'train',
            'validation',
            'test_public',
            'test_private',
            'test_private_mixed',
        }, f'unknown split: {split}'

        assert (
            mad2_object in MVTEC_AD_2_OBJECTS
        ), f'unknown MVTec AD 2 object: {mad2_object}'

        self.object = mad2_object
        self.split = split
        self.transform = transform

        self._image_base_dir = data_root

        self._object_dir = os.path.join(self._image_base_dir, mad2_object)
        # get all images from the split
        self._image_paths = sorted(glob.glob(self._get_pattern()))

    def _get_pattern(self) -> str:
        '''
        "[gb][oa][od]*" 這個小技巧是為了同時吃到 good/ 與 bad/ 兩種資料夾（g+o+o… = good；b+a+d… = bad）。
        private 類 split（test_private, test_private_mixed）不分 good/bad 資料夾，直接在該 split 底下就是一堆以數字開頭的檔名。
        self._image_paths：把上面 pattern 找到的路徑 排序後 存起來。__len__ 就是它的長度。
        '''
        if 'private' in self.split:
            # test_private / test_private_mixed：
            # <object>/<split>/<連號>.png
            return os.path.join(
                self._object_dir, self.split, '[0-9][0-9][0-9]*.png'
            )
            # train / validation / test_public：
            # <object>/<split>/<good|bad...>/<連號>.png
        return os.path.join(
            self._object_dir,
            self.split,
            '[gb][oa][od]*',
            '[0-9][0-9][0-9]*.png',
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item for the index ``idx``.

        Args:
            idx (int): Index to get the item.

        Returns:
            dict[str,  str | torch.Tensor]: Dict containing the sample image,
            image path, and the relative anomaly image output path for both
            image types continuous and thresholded.
        {
            'sample': Tensor(C,H,W),      # 影像張量（預設 0~1）
            'image_path': str,            # 原圖完整路徑
            'rel_out_path_cont': str,     # 之後輸出「連續」異常圖的相對路徑（.tiff）
            'rel_out_path_thresh': str,   # 之後輸出「二值」異常圖的相對路徑（.png）
        }
        這兩個「相對路徑」長這樣（會保留原本相對結構）：
        連續：anomaly_images/<object>/<split>/<good|bad>/<檔名>.tiff
        二值：anomaly_images_thresholded/<object>/<split>/<good|bad>/<檔名>.png
        未來你把模型輸出的 anomaly map 存檔時，直接用這兩個路徑接上你的實驗根目錄即可，評測伺服器要求的結構就對了。
        """

        image_path = self._image_paths[idx]
        sample = default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            'sample': sample, 
            'image_path': image_path,
            'rel_out_path_cont': self.get_relative_anomaly_image_out_path(idx),
            'rel_out_path_thresh': self.get_relative_anomaly_image_out_path(
                idx, True
            ),
        }

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def has_segmentation_gt(self) -> bool:
        return self.split == 'test_public'

    def get_relative_anomaly_image_out_path(self, idx, thresholded=False):
        """Returns a path relative to the experiment directory
        for storing the (thresholded) anomaly image in the required structure.

        Args:
            idx (int): sample index
            thresholded (bool): return output path for thresholded image,
            defaults to 'False'

        Returns:
            str: relative output path to write anomaly image
        """

        image_path = Path(self._image_paths[idx])
        relpath = image_path.relative_to(self._image_base_dir)

        if not thresholded:
            base_dir = 'anomaly_images'
            suffix = '.tiff'
        else:
            base_dir = 'anomaly_images_thresholded'
            suffix = '.png'

        return os.path.join(base_dir, relpath.with_suffix(suffix))

    def get_gt_image(self, idx):
        """Returns the ground truth image where values of 255 denote
        anomalous pixels and values of 0 anomaly-free ones. For good images
        'None' is returned.
        In case no segmentation ground truth is available
        (test_private/test_private_mixed) 'None' is returned as well.

        Args:
            idx (int): sample index

        Returns:
            numpy.array or None: ground truth image if available
            
        train/validation：通常只有正常（good）圖，且即便有 bad 也不會去抓 GT（因為屬性判斷只在 test_public 才為真）。
        test_public：
        good：get_gt_image(idx) 會回 None
        bad：回傳 H×W 的 numpy.array（255=異常, 0=正常）
        test_private / test_private_mixed：都回 None
        
        """
        gt_image = None
        if (
            self.has_segmentation_gt
            # 只對bad樣本回傳GT，good樣本回None
            # 路徑規則：把 ".../bad/<file>.png" 換成 ".../ground_truth/bad/<file>_mask.png"
            and 'good' not in self.get_relative_anomaly_image_out_path(idx)
        ):
            image_path = self.image_paths[idx]
            base_path, file_name = image_path.split('/bad/')
            gt_image_path = os.path.join(
                base_path, 'ground_truth/bad', file_name
            ).replace('.png', '_mask.png')

            gt_image_pil = Image.open(gt_image_path)
            gt_image = np.asarray(gt_image_pil)

        return gt_image
