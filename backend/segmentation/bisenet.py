import os
import cv2
import numpy as np
import torch
from torchvision import transforms

from .base import HairSegmenter

# Hair class in CelebAMask-HQ label map = 13
# Full label map: 0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
# 6=eye_g, 7=l_ear, 8=r_ear, 9=ear_r, 10=nose, 11=mouth, 12=u_lip, 13=hair,
# 14=hat, 15=l_lip, 16=cloth, 17=neck, 18=necklace
HAIR_LABEL = 13
GDRIVE_FILE_ID = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bisenet_79999_iter.pth")


class BiSeNetSegmenter(HairSegmenter):
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            print(f"Downloading BiSeNet weights to {MODEL_PATH}...")
            import gdown
            gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            print("Download complete.")

        self.model = self._build_bisenet()
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()

        h, w = image.shape[:2]

        img_resized = cv2.resize(image, (512, 512))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)[0]
            parsing = output.squeeze(0).argmax(0).cpu().numpy()

        hair_mask = (parsing == HAIR_LABEL).astype(np.uint8) * 255
        hair_mask = cv2.resize(hair_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        return hair_mask

    def _build_bisenet(self):
        """
        Build BiSeNet model from the vendored third-party source.
        Requires: git clone https://github.com/zllrunning/face-parsing.PyTorch
                  into backend/third_party/face_parsing/ (done in Step 4).
        """
        import sys
        third_party_path = os.path.join(os.path.dirname(__file__), "..", "third_party", "face_parsing")
        if third_party_path not in sys.path:
            sys.path.insert(0, third_party_path)
        from model import BiSeNet
        return BiSeNet(n_classes=19)
