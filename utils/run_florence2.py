import torch
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from transformers.dynamic_module_utils import get_imports
from typing import Optional, Union
from unittest.mock import patch

class Florence2Captioner:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = "microsoft/Florence-2-base", detail_level: int = 3):
        if self._is_initialized:  # 이미 초기화되었다면 스킵
            return
        
        self.model_name = model_name
        self.detail_level = detail_level
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        def fixed_get_imports(filename: str | Path) -> list[str]:
            imports = get_imports(filename)
            return [imp for imp in imports if imp != "flash_attn"] if str(filename).endswith("modeling_florence2.py") else imports
        
        self.model_path = Path("models") / model_name.replace('/', '_')
        if not self.model_path.exists():
            print(f"Downloading {model_name} model...")
            snapshot_download(repo_id=model_name, local_dir=self.model_path, local_dir_use_symlinks=False)
        
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
        
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self._is_initialized = True  # 초기화 완료 표시
        print("Model loaded and optimized.")
        
    def generate_caption(
        self,
        image_path: Union[str, Path],
        num_beams: int = 3,
        max_new_tokens: int = 1024,
        prepend_text: str = "",
        append_text: str = ""
    ) -> str:
        if type(image_path) == str:
            image = Image.open(image_path).convert("RGB")
        
        prompt = {
            1: '<CAPTION>',
            2: '<DETAILED_CAPTION>',
            3: '<MORE_DETAILED_CAPTION>'
        }.get(self.detail_level, '<MORE_DETAILED_CAPTION>')

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            do_rescale=False
        )

        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device, torch.bfloat16)
        }
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams
            )
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        caption = caption.replace('</s>', '').replace('<s>', '').replace('<pad>', '')
        
        return f"{prepend_text}{caption}{append_text}".strip()
    
    def save_caption(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> str:
        caption = self.generate_caption(image_path, **kwargs)
        
        if output_path is None:
            output_path = Path(image_path).with_suffix('.txt')
        
        Path(output_path).write_text(caption)
        return caption

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