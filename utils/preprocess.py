"""
Dermoscopic image preprocessing utilities.

1. Hair artifact removal  — DullRazor-inspired (morphological + inpainting)
2. Color constancy        — Shades of Gray algorithm
"""

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Hair removal
# ─────────────────────────────────────────────────────────────────────────────

def remove_hair(image: np.ndarray, kernel_size: int = 17,
                threshold: int = 10) -> np.ndarray:
    """
    Remove hair artifacts from a dermoscopic image using morphological
    bottom-hat filtering followed by bilinear inpainting.

    Args:
        image       : BGR uint8 numpy array (H, W, 3)
        kernel_size : Size of the linear structuring element
        threshold   : Intensity threshold to create the hair mask

    Returns:
        Cleaned BGR uint8 numpy array
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bottom-hat filter: detects dark, thin structures (hair)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create binary hair mask
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # Inpaint over the hair mask
    cleaned = cv2.inpaint(image, mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Color constancy
# ─────────────────────────────────────────────────────────────────────────────

def shades_of_gray(image: np.ndarray, power: float = 6.0) -> np.ndarray:
    """
    Shades of Gray color constancy algorithm.
    Normalizes illumination to reduce color cast from different dermoscopes.

    Args:
        image : BGR uint8 numpy array (H, W, 3)
        power : Minkowski norm order (6 approximates max-RGB)

    Returns:
        Color-corrected BGR uint8 numpy array
    """
    img_float = image.astype(np.float32) + 1e-6

    # Compute Minkowski norm per channel
    norm = np.power(
        np.mean(np.power(img_float, power), axis=(0, 1)),
        1.0 / power
    )

    # Scale channels so the norm equals the overall mean
    scale = np.mean(norm) / (norm + 1e-6)
    corrected = np.clip(img_float * scale, 0, 255).astype(np.uint8)
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(pil_image: Image.Image,
                     remove_hair_flag: bool = True,
                     color_constancy_flag: bool = True) -> Image.Image:
    """
    Full preprocessing pipeline for a single dermoscopic PIL image.

    Args:
        pil_image           : Input PIL RGB image
        remove_hair_flag    : Whether to apply hair removal
        color_constancy_flag: Whether to apply color constancy

    Returns:
        Preprocessed PIL RGB image
    """
    # PIL (RGB) → OpenCV (BGR)
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if remove_hair_flag:
        img_bgr = remove_hair(img_bgr)

    if color_constancy_flag:
        img_bgr = shades_of_gray(img_bgr)

    # BGR → RGB → PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
