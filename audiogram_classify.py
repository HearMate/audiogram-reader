import cv2
import numpy as np
import os


def classify_audiogram_type(image_path: str) -> str:
    """
    Classifies the audiogram image into one of two types:
    - 'type_1': Labeled manually with Polish text ("ucho prawe", etc.)
    - 'type_2': Clinical/automatically generated audiogram

    Parameters:
        image_path (str): Path to the audiogram image file.

    Returns:
        str: 'type_1', 'type_2', or 'unknown'
    """
    filename = os.path.basename(image_path).lower()
    if any(keyword in filename for keyword in ["ucho", "prawe", "lewe"]):
        print("[INFO] Polish keywords detected in filename → classified as type_1")
        return "type_1"

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found at the provided path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print(f"[INFO] Mean brightness: {brightness:.2f}")

    gray_pixels = np.logical_and(gray > 80, gray < 180)
    gray_ratio = np.sum(gray_pixels) / gray.size
    print(f"[INFO] Gray pixel ratio: {gray_ratio:.2f}")

    if gray_ratio > 0.3:
        print("[INFO] High gray ratio detected → classified as type_2")
        return "type_2"
    elif brightness > 180 and gray_ratio < 0.2:
        print("[INFO] Bright and clean image → classified as type_1")
        return "type_1"
    else:
        print("[INFO] Unclear structure → classified as unknown")
        return "unknown"
