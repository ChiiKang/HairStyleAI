import cv2
import numpy as np
import torch

from .base import HairSegmenter

# jonathandinu/face-parsing uses CelebAMask-HQ labels -- hair = 13
# (17 = neck, not hair)
HAIR_LABEL = 13
MODEL_NAME = "jonathandinu/face-parsing"


class SegFormerSegmenter(HairSegmenter):
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        h, w = image.shape[:2]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        parsing = upsampled.argmax(dim=1).squeeze(0).cpu().numpy()

        hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255

        return hair_mask
