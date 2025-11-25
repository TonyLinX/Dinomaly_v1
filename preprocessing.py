import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def my_preprocessing(pil_img: Image.Image) -> Image.Image:
    """
    光照正規化 (Illumination Normalization)：
    以大型 Gaussian Blur 估計光照，再做除法補償，減少反光與陰影。
    """
    img_np = np.array(pil_img)
    blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=15, sigmaY=15)

    img_float = img_np.astype(np.float32)
    blur_float = blur.astype(np.float32)
    epsilon = 1.0
    normalized = img_float / (blur_float + epsilon) * 128.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return Image.fromarray(normalized)


def attach_preprocessing(transform):
    """
    把 my_preprocessing 插入特定 transform pipeline 的最前面。
    transform: torchvision.transforms.Compose or any callable.
    """
    if transform is None:
        raise ValueError("transform must not be None when attaching preprocessing.")
    return transforms.Compose([
        transforms.Lambda(my_preprocessing),
        transform,
    ])
