import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.run_fsrgan import Generator

class SuperResolutionProcessor:
    def __init__(self, min_process_size=256, max_process_size=512):
        self.min_process_size = min_process_size
        self.max_process_size = max_process_size
        
        device = "cuda" if torch.cuda.is_available() else "cpu" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

        config = OmegaConf.load("config/fsrgan_config.yaml")
        self.model = Generator(config.generator)
        weights = torch.load("models/fsrgan_model.pt", map_location=device)

        new_weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
        self.model.load_state_dict(new_weights)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def normalize_size(self, img):
        h, w = img.shape[:2]
        original_ratio = h / w

        max_side = max(h, w)
        if max_side <= self.min_process_size:
            target_size = self.min_process_size
            interpolation = cv2.INTER_CUBIC 
        else:
            target_size = min(self.max_process_size, max_side)
            interpolation = cv2.INTER_AREA 
        
        resized = cv2.resize(img, (target_size, target_size), interpolation=interpolation)
        
        return resized, original_ratio, (h, w)

    def restore_size(self, img, original_ratio, original_size):
        h, w = original_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_image(self, img):
        normalized_img, original_ratio, original_size = self.normalize_size(img)
        
        rgb_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)
        
        lr_image = (torch.from_numpy(rgb_img) / 127.5) - 1.0
        lr_image = lr_image.permute(2, 0, 1).unsqueeze(dim=0).to(self.device)
        
        with torch.no_grad():
            sr_image = self.model(lr_image).cpu()
            sr_image = (sr_image + 1.0) / 2.0
            sr_image = sr_image.permute(0, 2, 3, 1).squeeze()
            sr_image = (sr_image * 255).numpy().astype(np.uint8)
        
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        restored_img = self.restore_size(sr_image, original_ratio, original_size)
        
        return restored_img