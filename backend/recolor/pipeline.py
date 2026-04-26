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


def _apply_lift(lab_image: np.ndarray, mask_float: np.ndarray, lift: int) -> None:
    """Apply brightness lift to dark hair pixels in-place."""
    if lift <= 0:
        return
    lift_amount = lift * 2.55
    l_channel = lab_image[:, :, 0]
    dark_mask = (l_channel < 100).astype(np.float32)
    lab_image[:, :, 0] = np.clip(
        l_channel + lift_amount * dark_mask * mask_float, 0, 255
    )


def _recolor_overlay(
    lab_image: np.ndarray, lab_target: np.ndarray,
    mask_float: np.ndarray, alpha: float,
) -> None:
    """Original method: absolute A/B replacement (flat, uniform)."""
    blend = alpha * mask_float
    for ch in [1, 2]:
        lab_image[:, :, ch] = (
            lab_image[:, :, ch] * (1 - blend)
            + lab_target[:, :, ch] * blend
        )


def _recolor_shift(
    lab_image: np.ndarray, lab_target: np.ndarray,
    mask_float: np.ndarray, alpha: float,
) -> None:
    """Relative color shift: move mean A/B toward target, preserving variation."""
    hair_mask_bool = mask_float > 0.5
    if not np.any(hair_mask_bool):
        return

    for ch in [1, 2]:
        hair_pixels = lab_image[:, :, ch][hair_mask_bool]
        mean_hair = hair_pixels.mean()
        target_val = lab_target[0, 0, ch]
        shift = (target_val - mean_hair) * alpha
        lab_image[:, :, ch] += shift * mask_float


def _recolor_reinhard(
    lab_image: np.ndarray, lab_target: np.ndarray,
    mask_float: np.ndarray, alpha: float,
) -> None:
    """
    Reinhard color transfer: match mean AND std deviation of target color.
    Preserves the natural highlight/shadow variation in hair.
    Based on Reinhard et al. "Color Transfer between Images" (2001).
    """
    hair_mask_bool = mask_float > 0.5
    if not np.any(hair_mask_bool):
        return

    target_a = lab_target[0, 0, 1]
    target_b = lab_target[0, 0, 2]

    for ch in [1, 2]:
        hair_pixels = lab_image[:, :, ch][hair_mask_bool]
        src_mean = hair_pixels.mean()
        src_std = hair_pixels.std()
        if src_std < 1e-6:
            src_std = 1.0

        target_val = target_a if ch == 1 else target_b
        # Use a moderate target std to preserve variation without flattening
        target_std = max(src_std * 0.7, 5.0)

        # Reinhard: normalize, scale to target std, shift to target mean
        transferred = (lab_image[:, :, ch] - src_mean) * (target_std / src_std) + target_val

        # Blend with original using mask and intensity
        lab_image[:, :, ch] = (
            lab_image[:, :, ch] * (1 - alpha * mask_float)
            + transferred * alpha * mask_float
        )


RECOLOR_METHODS = {
    "overlay": _recolor_overlay,
    "shift": _recolor_shift,
    "reinhard": _recolor_reinhard,
}


def recolor_hair(
    image: np.ndarray,
    mask: np.ndarray,
    color_hex: str,
    intensity: int = 80,
    lift: int = 0,
    method: str = "reinhard",
) -> np.ndarray:
    if intensity == 0:
        return image.copy()

    alpha = intensity / 100.0
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    target_bgr = np.full((1, 1, 3), hex_to_bgr(color_hex), dtype=np.uint8)
    lab_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    mask_float = mask.astype(np.float32) / 255.0

    _apply_lift(lab_image, mask_float, lift)

    recolor_fn = RECOLOR_METHODS.get(method, _recolor_reinhard)
    recolor_fn(lab_image, lab_target, mask_float, alpha)

    lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Restore original pixels where mask is zero to avoid LAB round-trip artifacts
    zero_mask = (mask == 0)
    result[zero_mask] = image[zero_mask]

    return result
