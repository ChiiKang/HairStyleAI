import numpy as np
import cv2
import pytest
from recolor.pipeline import clean_mask, recolor_hair


# --- Mask cleanup tests ---


def test_clean_mask_removes_small_islands():
    """Small isolated blobs (< 100px) should be removed."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    mask[80:83, 80:83] = 255

    cleaned = clean_mask(mask, min_area=100)

    assert cleaned[40, 40] == 255, "Large region should survive"
    assert cleaned[81, 81] == 0, "Small island should be removed"


def test_clean_mask_fills_small_holes():
    """Small holes inside the mask should be filled."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    mask[38:41, 38:41] = 0

    cleaned = clean_mask(mask, min_area=100)

    assert cleaned[39, 39] == 255, "Small hole should be filled"


# --- Shared recolor tests (all methods) ---


def _make_hair_image():
    """Create a test image with color variation to simulate real hair."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Base brown hair color with variation
    for y in range(100):
        for x in range(100):
            # Vary brightness to simulate strands
            v = 60 + (y % 10) * 3 + (x % 7) * 2
            img[y, x] = [int(v * 0.5), int(v * 0.7), v]  # BGR brownish
    return img


def _make_varied_mask():
    """Mask with full hair region."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:90] = 255
    return mask


@pytest.mark.parametrize("method", ["reinhard", "shift", "overlay"])
def test_recolor_preserves_unmasked_areas(method):
    image = _make_hair_image()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0, method=method)

    np.testing.assert_array_equal(result[0, 0], image[0, 0])


@pytest.mark.parametrize("method", ["reinhard", "shift", "overlay"])
def test_recolor_changes_masked_area(method):
    image = _make_hair_image()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0, method=method)

    assert not np.array_equal(result[50, 50], image[50, 50])


@pytest.mark.parametrize("method", ["reinhard", "shift", "overlay"])
def test_recolor_zero_intensity_is_noop(method):
    image = _make_hair_image()
    mask = np.full((100, 100), 255, dtype=np.uint8)

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=0, lift=0, method=method)

    np.testing.assert_array_equal(result, image)


# --- Reinhard-specific: preserves color variation ---


def test_reinhard_preserves_hair_variation():
    """Reinhard should preserve pixel-to-pixel variation, not flatten to uniform color."""
    image = _make_hair_image()
    mask = _make_varied_mask()

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=80, lift=0, method="reinhard")

    # Check variation in LAB space (avoids uint8 rounding in BGR)
    hair_region = result[20:80, 20:80]
    lab_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2LAB).astype(float)
    # L channel should have variation (brightness differences preserved)
    std_l = lab_region[:, :, 0].std()
    assert std_l > 0.5, f"Reinhard should preserve luminance variation, got std_L={std_l}"


def test_overlay_flattens_more_than_reinhard():
    """Overlay should produce less variation than Reinhard at same intensity."""
    image = _make_hair_image()
    mask = _make_varied_mask()

    result_reinhard = recolor_hair(image, mask, "#FF0000", intensity=80, method="reinhard")
    result_overlay = recolor_hair(image, mask, "#FF0000", intensity=80, method="overlay")

    # Measure variation in the hair region
    def hair_std(img):
        region = img[20:80, 20:80]
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(float)
        return lab[:, :, 1].std() + lab[:, :, 2].std()

    reinhard_var = hair_std(result_reinhard)
    overlay_var = hair_std(result_overlay)

    assert reinhard_var > overlay_var, (
        f"Reinhard should preserve more variation ({reinhard_var:.1f}) than overlay ({overlay_var:.1f})"
    )


def test_shift_preserves_relative_differences():
    """Shift should keep lighter pixels lighter and darker pixels darker."""
    image = _make_hair_image()
    mask = _make_varied_mask()

    result = recolor_hair(image, mask, "#3498DB", intensity=80, method="shift")

    # Compare two points that had different brightness in original
    orig_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(float)

    # Point at (30,30) vs (35,35) — should maintain relative A/B ordering
    # The shift should move them by the same amount
    orig_diff_a = orig_lab[30, 30, 1] - orig_lab[35, 35, 1]
    result_diff_a = result_lab[30, 30, 1] - result_lab[35, 35, 1]

    # The relative difference should be roughly preserved (within tolerance for uint8 rounding)
    assert abs(orig_diff_a - result_diff_a) < 10.0, (
        f"Shift should preserve relative differences: orig={orig_diff_a:.1f}, result={result_diff_a:.1f}"
    )


def test_recolor_with_lift():
    """Lift should brighten dark pixels in the masked area."""
    image = np.full((100, 100, 3), [30, 20, 15], dtype=np.uint8)  # Very dark
    mask = np.full((100, 100), 255, dtype=np.uint8)

    result_no_lift = recolor_hair(image, mask, "#FF0000", intensity=80, lift=0, method="reinhard")
    result_lift = recolor_hair(image, mask, "#FF0000", intensity=80, lift=30, method="reinhard")

    # Lifted version should be brighter
    lab_no = cv2.cvtColor(result_no_lift, cv2.COLOR_BGR2LAB).astype(float)
    lab_lift = cv2.cvtColor(result_lift, cv2.COLOR_BGR2LAB).astype(float)

    mean_l_no = lab_no[50, 50, 0]
    mean_l_lift = lab_lift[50, 50, 0]

    assert mean_l_lift > mean_l_no, f"Lift should brighten: no_lift L={mean_l_no:.1f}, lift L={mean_l_lift:.1f}"
