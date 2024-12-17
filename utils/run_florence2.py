import torch
from PIL import Image
from pathlib import Path
from unittest.mock import patch
from typing import Optional, Union
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_imports
from transformers import AutoModelForCausalLM, AutoProcessor

class Florence2Captioner:
    # Singleton instance and initialization flag
    # 싱글톤 인스턴스와 초기화 플래그
    _instance = None
    _is_initialized = False

    # Singleton pattern implementation
    # 싱글톤 패턴 구현 이렇게 하면 클래스 생성 시 어떤 인자가 전달되더라도 모두 수용할 수 있음
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Initialize the Florence2 model with specified parameters
    # 지정된 매개변수로 Florence2 모델 초기화
    def __init__(self, model_name: str = "microsoft/Florence-2-base", detail_level: int = 3):
        if self._is_initialized:
            return
        
        self.model_name = model_name
        self.detail_level = detail_level
        
        # Set device to GPU if available, otherwise CPU
        # GPU 사용 가능시 GPU 사용, 아니면 CPU 사용
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Fix imports to handle flash attention compatibility
        # 플래시 어텐션 호환성을 위한 임포트 수정
        def fixed_get_imports(filename: str | Path) -> list[str]:
            imports = get_imports(filename)
            return [imp for imp in imports if imp != "flash_attn"] if str(filename).endswith("modeling_florence2.py") else imports
        
        # Download and prepare model if not already present
        # 모델이 없는 경우 다운로드 및 준비
        self.model_path = Path("models") / model_name.replace('/', '_')
        if not self.model_path.exists():
            print(f"Downloading {model_name} model...")
            snapshot_download(repo_id=model_name, local_dir=self.model_path, local_dir_use_symlinks=False)
        
        # Load model and processor with optimized settings
        # 최적화된 설정으로 모델과 프로세서 로드
        print(f"Loading model on {self.device}...")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        
        # Compile model for performance optimization
        # 성능 최적화를 위한 모델 컴파일
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self._is_initialized = True
        print("Model loaded and optimized.")
        
    # Generate caption for the given image
    # 주어진 이미지에 대한 캡션 생성
    def generate_caption(self, image_path: Union[str, Path], num_beams: int = 3, max_new_tokens: int = 1024, prepend_text: str = "", append_text: str = "") -> str:
        # Load and convert image to RGB
        # 이미지를 로드하고 RGB로 변환
        if type(image_path) == str:
            image = Image.open(image_path).convert("RGB")
        
        # Select prompt based on detail level
        # 상세도 수준에 따라 프롬프트 선택
        prompt = {
            1: '<CAPTION>',
            2: '<DETAILED_CAPTION>',
            3: '<MORE_DETAILED_CAPTION>'
        }.get(self.detail_level, '<MORE_DETAILED_CAPTION>')

        # Process image and text inputs
        # 이미지와 텍스트 입력 처리
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            do_rescale=False
        )

        # Prepare inputs for model
        # 모델용 입력 준비
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device, torch.bfloat16)
        }
        
        # Generate caption using the model
        # 모델을 사용하여 캡션 생성
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams
            )
        
        # Process and clean up the generated caption
        # 생성된 캡션 처리 및 정리
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        caption = caption.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
        
        return f"{prepend_text}{caption}{append_text}".strip()
    
    # Save generated caption to file
    # 생성된 캡션을 파일로 저장
    def save_caption(self, image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, **kwargs) -> str:
        caption = self.generate_caption(image_path, **kwargs)
        
        # If no output path specified, use image path with .txt extension
        # 출력 경로가 지정되지 않은 경우 이미지 경로에 .txt 확장자 사용
        if output_path is None:
            output_path = Path(image_path).with_suffix('.txt')
        
        Path(output_path).write_text(caption)
        return caption

# Main execution block for testing
# 테스트를 위한 메인 실행 블록
if __name__ == "__main__":
    captioner = Florence2Captioner(
        model_name="microsoft/Florence-2-large",
        detail_level=3
    )
    
    image_path = "../Super-Resolution/data/son.jpg"
    try:
        caption = captioner.generate_caption(
            image_path,
            prepend_text="Description: ",
            append_text=" [End of description]"
        )
        print(f"Generated caption: {caption}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")