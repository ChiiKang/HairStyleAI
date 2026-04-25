from abc import ABC, abstractmethod
import numpy as np


class HairSegmenter(ABC):
    """Abstract base for all hair segmentation models."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Called once at startup."""
        ...

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment hair from an image.

        Args:
            image: BGR image (OpenCV format), any size.

        Returns:
            Grayscale mask (0-255), same height/width as input.
            255 = hair, 0 = not hair.
        """
        ...
