"""
(A) Global exposure aug（模擬 overexposed / underexposed）

目的：讓 g 學會「整張比較亮 / 比較暗」時 representation 仍然靠近 regular。
可以做很溫和就好，range 大概對齊你這幾張圖的差距：
brightness factor ∈ [0.9, 1.12]
gamma factor ∈ [0.9, 1.1]

這一類就對應你 dataset 的：
overexposed ≈ brightness ↑ 一點
underexposed ≈ brightness ↓ 一點

Pseudo-code（用 Albumentations 為例）：
"""

import albumentations as A

global_exposure_aug = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(-0.10, 0.12),  # 大概 ±10%
        contrast_limit=(-0.05, 0.08),
        p=1.0
    ),
    A.RandomGamma(gamma_limit=(90, 110), p=0.5),  # gamma ≈ 0.9~1.1
])


"""
(B) Directional band light / shadow（模擬 shift_1 / shift_2 / shift_3）

目的：專門模擬「一條亮帶 / 暗帶沿著罐子移動」的現象，
也就是你最痛的 局部反光。

做法：在影像上乘上一個 平滑的一維 mask，只改「亮度」，不亂改幾何。

對這個 can 類別來說，罐子是橫躺的，所以亮帶是 沿著長邊方向延伸，
也就是「對 row 做 Gaussian band」，column 幾乎不變。

你可以把它包成 Albumentations 的 ImageOnlyTransform，叫 BandLightShift。
這個 transform 自然會產生很多種情況：

band 比較靠下面 → 很像 shift_1
band 靠中間 → 很像 shift_2
band 靠上面 + 強一點 → 很像 shift_3

k < 0 → 暗帶
k > 0 → 亮帶（強反光）

這樣一個 family 就 cover 了你 3 種 shift，甚至還多了一些你現在 test set 沒有的情況（對 generalization 反而是好事）。

簡化版的 transform：
"""

import numpy as np

def band_light_shift(img,
                     strength_range=(-0.25, 0.4),
                     center_range=(0.25, 0.75),
                     width_range=(0.15, 0.35)):
    """
    img: HxWxC (uint8)
    strength_range: band 強度 k，>0 變亮，<0 變暗
    center_range: band 中心在高度方向的位置比例
    width_range: band 寬度 (相對於高度)
    """
    h, w, c = img.shape
    y = np.arange(h).reshape(h, 1).astype(np.float32)

    center = np.random.uniform(*center_range) * h
    width  = np.random.uniform(*width_range) * h

    # Gaussian band in vertical direction
    band = np.exp(-0.5 * ((y - center) / width) ** 2)  # [H, 1]

    k = np.random.uniform(*strength_range)             # 亮或暗帶
    mask = 1.0 + k * band                              # [H, 1]

    img_f = img.astype(np.float32)
    img_f *= mask[..., None]                           # broadcast 到 C
    img_f = np.clip(img_f, 0, 255).astype(np.uint8)
    return img_f

def apply_light_augment(img_np):
    """
    輸入一張原圖 (HxWxC, uint8)，
    先做 global_exposure_aug，再做 band_light_shift，
    回傳一張 augment 後的圖片。
    """
    # global 曝光
    aug = global_exposure_aug(image=img_np)["image"]
    # band 亮帶/暗帶
    aug = band_light_shift(aug)
    return aug