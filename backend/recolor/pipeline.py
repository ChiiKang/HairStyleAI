import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    min_area: int = 100,
    feather_radius: int = 5,
) -> np.ndarray:
    """Clean a binary mask: remove small islands, fill small holes, feather edges."""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    inverted = cv2.bitwise_not(cleaned)
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inverted)
    for i in range(1, num_labels_inv):
        if stats_inv[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels_inv == i] = 255

    if feather_radius > 0:
        k = feather_radius * 2 + 1
        cleaned = cv2.GaussianBlur(cleaned, (k, k), 0)

    return cleaned


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (B, G, R)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def recolor_hair(
    image: np.ndarray,
    mask: np.ndarray,
    color_hex: str,
    intensity: int = 80,
    lift: int = 0,
) -> np.ndarray:
    if intensity == 0:
        return image.copy()

    alpha = intensity / 100.0

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    target_bgr = np.full_like(image, hex_to_bgr(color_hex), dtype=np.uint8)
    lab_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_float = mask.astype(np.float32) / 255.0
    blend = alpha * mask_float

    if lift > 0:
        lift_amount = lift * 2.55
        l_channel = lab_image[:, :, 0]
        dark_mask = (l_channel < 100).astype(np.float32)
        l_lifted = l_channel + lift_amount * dark_mask * mask_float
        lab_image[:, :, 0] = np.clip(l_lifted, 0, 255)

    for ch in [1, 2]:
        lab_image[:, :, ch] = (
            lab_image[:, :, ch] * (1 - blend)
            + lab_target[:, :, ch] * blend
        )

    lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Restore original pixels where mask is fully zero to avoid LAB round-trip rounding artifacts
    zero_mask = (mask == 0)
    result[zero_mask] = image[zero_mask]

    return result
