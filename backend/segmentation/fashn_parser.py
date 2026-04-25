import cv2
import numpy as np

from .base import HairSegmenter

# FASHN Human Parser uses ATR label convention: hair = 2
# Verify at runtime: print(parser.predict(img)) on a test image
# If label is wrong, check model card: https://huggingface.co/fashn-ai/fashn-human-parser
HAIR_LABEL = 2


class FASHNSegmenter(HairSegmenter):
    def __init__(self):
        self.parser = None

    def load(self) -> None:
        try:
            from fashn_human_parser import FashnHumanParser
            self.parser = FashnHumanParser()
        except ImportError as e:
            raise ImportError(
                "fashn-human-parser is not installed or could not be imported. "
                "Install with: pip install fashn-human-parser\n"
                f"Original error: {e}"
            )

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.parser is None:
            self.load()

        h, w = image.shape[:2]

        from PIL import Image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)

        seg_map = self.parser.predict(pil_image)

        hair_mask = (seg_map == HAIR_LABEL).astype(np.uint8) * 255

        if hair_mask.shape[:2] != (h, w):
            hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return hair_mask
