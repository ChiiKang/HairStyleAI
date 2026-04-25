import numpy as np
from recolor.pipeline import clean_mask


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


from recolor.pipeline import recolor_hair


def test_recolor_preserves_unmasked_areas():
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0)

    np.testing.assert_array_equal(result[0, 0], image[0, 0])


def test_recolor_changes_masked_area():
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=100, lift=0)

    assert not np.array_equal(result[50, 50], image[50, 50])


def test_recolor_zero_intensity_is_noop():
    image = np.full((100, 100, 3), [120, 80, 60], dtype=np.uint8)
    mask = np.full((100, 100), 255, dtype=np.uint8)

    result = recolor_hair(image, mask, color_hex="#FF0000", intensity=0, lift=0)

    np.testing.assert_array_equal(result, image)
