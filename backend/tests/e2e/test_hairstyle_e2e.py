"""
End-to-end Playwright test for hairstyle generation.

Requires:
- Backend running on :8000 with FAL_KEY set
- Frontend running on :5173
- A test image

Run: pytest tests/e2e/test_hairstyle_e2e.py -v --headed
"""

import os
import numpy as np
import cv2
import pytest
from playwright.sync_api import Page, expect


# Skip if FAL_KEY not set (can't test generation without it)
pytestmark = pytest.mark.skipif(
    not os.environ.get("FAL_KEY"),
    reason="FAL_KEY not set — skipping E2E generation test",
)

TEST_IMAGE_PATH = "/tmp/test_selfie_e2e.png"


@pytest.fixture(scope="session", autouse=True)
def create_test_image():
    """Create a simple test portrait image."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a simple face-like shape (skin tone oval)
    cv2.ellipse(img, (256, 220), (120, 160), 0, 0, 360, (120, 160, 200), -1)
    # Hair region on top
    cv2.ellipse(img, (256, 160), (140, 120), 0, 180, 360, (40, 30, 25), -1)
    # Eyes
    cv2.circle(img, (210, 210), 12, (60, 40, 30), -1)
    cv2.circle(img, (300, 210), 12, (60, 40, 30), -1)
    cv2.imwrite(TEST_IMAGE_PATH, img)
    yield
    os.unlink(TEST_IMAGE_PATH)


def test_hairstyle_tab_visible(page: Page):
    """Tab bar shows 'Hairstyle Generator' option."""
    page.goto("http://localhost:5173")
    tab = page.get_by_text("Hairstyle Generator")
    expect(tab).to_be_visible()


def test_hairstyle_tab_shows_upload(page: Page):
    """Clicking the tab shows the selfie upload area."""
    page.goto("http://localhost:5173")
    page.get_by_text("Hairstyle Generator").click()
    expect(page.get_by_text("Upload a photo or take a selfie")).to_be_visible()
    expect(page.get_by_text("Generation Model")).to_be_visible()
    expect(page.get_by_text("Generate Hairstyles")).to_be_visible()


def test_upload_image_shows_preview(page: Page):
    """Uploading an image shows the preview."""
    page.goto("http://localhost:5173")
    page.get_by_text("Hairstyle Generator").click()

    # Upload via file input
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(TEST_IMAGE_PATH)

    # Should show the preview image
    preview = page.locator("img[alt='Selected']")
    expect(preview).to_be_visible(timeout=5000)


def test_model_selector_has_options(page: Page):
    """Model selector contains all 4 fal.ai models."""
    page.goto("http://localhost:5173")
    page.get_by_text("Hairstyle Generator").click()

    selector = page.locator("select").last
    options = selector.locator("option").all_text_contents()
    assert any("GPT Image 2" in o for o in options)
    assert any("Luma Photon" in o for o in options)
    assert any("Qwen" in o for o in options)
    assert any("Seedream" in o for o in options)


def test_generate_hairstyles_with_cheapest_model(page: Page):
    """
    Full E2E: upload image, select cheapest model (Luma Photon Flash),
    generate, and verify 4 result images appear.
    """
    page.goto("http://localhost:5173")
    page.get_by_text("Hairstyle Generator").click()

    # Upload test image
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(TEST_IMAGE_PATH)
    expect(page.locator("img[alt='Selected']")).to_be_visible(timeout=5000)

    # Select cheapest model (Luma Photon Flash)
    selector = page.locator("select").last
    selector.select_option(value="fal-ai/luma-photon/flash")

    # Click generate
    page.get_by_text("Generate Hairstyles").click()

    # Wait for loading spinners to appear
    expect(page.get_by_text("Generating style 1...")).to_be_visible(timeout=5000)

    # Wait for results — generation can take up to 60 seconds
    # Look for the "Save" download links that appear when images load
    save_links = page.get_by_text("Save")
    expect(save_links.first).to_be_visible(timeout=120000)

    # Verify we got 4 result images
    result_images = page.locator("img[alt='Protective Braids'], img[alt='Natural Twist-Out'], img[alt='Silk Press'], img[alt='Bantu Knots']")
    expect(result_images).to_have_count(4, timeout=120000)

    # Verify model name and duration shown (e.g. "flash — 9.6s")
    duration_label = page.locator("span.text-xs.text-gray-500")
    expect(duration_label).to_be_visible()
    assert "flash" in duration_label.text_content()
