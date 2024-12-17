import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from utils.run_fsrgan import Generator

class SuperResolutionProcessor:
    # Initialize the processor with size constraints
    # 크기 제한을 가진 프로세서 초기화
    def __init__(self, min_process_size=256, max_process_size=512):
        self.min_process_size = min_process_size
        self.max_process_size = max_process_size
        
        # Determine the appropriate device (CUDA GPU, MPS, or CPU)
        # 적절한 디바이스 결정 (CUDA GPU, MPS, 또는 CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

        # Load model configuration and initialize generator
        # 모델 설정을 로드하고 생성자 초기화
        config = OmegaConf.load("config/fsrgan_config.yaml")
        self.model = Generator(config.generator)
        weights = torch.load("models/fsrgan_model.pt", map_location=device)

        # Clean up weight keys and load into model
        # 가중치 키를 정리하고 모델에 로드
        new_weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
        self.model.load_state_dict(new_weights)
        self.model.to(device)
        self.model.eval()
        self.device = device

    # Normalize image size while maintaining aspect ratio
    # 종횡비를 유지하면서 이미지 크기 정규화
    def normalize_size(self, img):
        h, w = img.shape[:2]
        original_ratio = h / w

        # Determine target size and interpolation method based on image dimensions
        # 이미지 크기에 따라 목표 크기와 보간 방법 결정
        max_side = max(h, w)
        if max_side <= self.min_process_size:
            target_size = self.min_process_size
            # Use cubic interpolation for upscaling
            # 업스케일링에는 큐빅 보간법 사용
            interpolation = cv2.INTER_CUBIC
        else:
            target_size = min(self.max_process_size, max_side)
            # Use area interpolation for downscaling
            # 다운스케일링에는 영역 보간법 사용
            interpolation = cv2.INTER_AREA
        
        # Resize image to target size
        # 이미지를 목표 크기로 리사이즈
        resized = cv2.resize(img, (target_size, target_size), interpolation=interpolation)
        
        return resized, original_ratio, (h, w)

    # Restore image to original size
    # 이미지를 원래 크기로 복원
    def restore_size(self, img, original_ratio, original_size):
        h, w = original_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Main processing function for super-resolution
    # 초해상도를 위한 주요 처리 함수
    def process_image(self, img):
        # Normalize input image size
        # 입력 이미지 크기 정규화
        normalized_img, original_ratio, original_size = self.normalize_size(img)
        
        # Convert BGR to RGB color space
        # BGR에서 RGB 색공간으로 변환
        rgb_img = cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB)
        
        # Prepare image for model input (normalize to [-1, 1] range)
        # 모델 입력을 위한 이미지 준비 ([-1, 1] 범위로 정규화)
        lr_image = (torch.from_numpy(rgb_img) / 127.5) - 1.0
        lr_image = lr_image.permute(2, 0, 1).unsqueeze(dim=0).to(self.device)
        
        # Process image through model
        # 모델을 통한 이미지 처리
        with torch.no_grad():
            # Generate super-resolution image
            # 초해상도 이미지 생성
            sr_image = self.model(lr_image).cpu()
            # Convert from [-1, 1] to [0, 1] range
            # [-1, 1] 범위에서 [0, 1] 범위로 변환
            sr_image = (sr_image + 1.0) / 2.0
            # Rearrange dimensions and convert to numpy array
            # 차원 재배열 및 numpy 배열로 변환
            sr_image = sr_image.permute(0, 2, 3, 1).squeeze()
            sr_image = (sr_image * 255).numpy().astype(np.uint8)
        
        # Convert back to BGR color space
        # RGB에서 BGR 색공간으로 다시 변환
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        
        # Restore to original image size
        # 원래 이미지 크기로 복원
        restored_img = self.restore_size(sr_image, original_ratio, original_size)
        
        return restored_img