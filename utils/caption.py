from threading import Thread, Lock
from utils.run_florence2 import Florence2Captioner

captioner = Florence2Captioner(model_name="microsoft/Florence-2-base", detail_level=3)

class CaptionState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionState, cls).__new__(cls)
            cls._instance.current_caption = "Processing caption..."
            cls._instance.is_processing = False
            cls._instance.lock = Lock()
        return cls._instance

    def get_state(self):
        with self.lock:
            return {
                'caption': self.current_caption,
                'is_processing': self.is_processing
            }
    
    def start_processing(self):
        with self.lock:
            self.is_processing = True
            self.current_caption = "Processing caption..."
            print("Started processing caption")
    
    def finish_processing(self, caption):
        with self.lock:
            self.current_caption = caption
            self.is_processing = False
            print(f"Caption generated successfully: {caption[:50]}...")

caption_state = CaptionState()
def generate_caption_async(image_path):
    def _generate():
        try:
            print(f"Starting caption generation for {image_path}")
            print("Generating caption...")
            caption = captioner.generate_caption(image_path)
            print("Caption generated, updating state...")
            caption_state.finish_processing(caption)
        except Exception as e:
            print(f"Caption generation error: {e}")
            import traceback
            print(traceback.format_exc())
            caption_state.finish_processing(f"Failed to generate caption: {str(e)}")
    
    thread = Thread(target=_generate)
    thread.daemon = True
    thread.start()
    return thread