from threading import Thread, Lock
from utils.run_florence2 import Florence2Captioner

# Initialize the Florence2 captioner with specified settings
# Florence2 캡션 생성기를 지정된 설정으로 초기화
captioner = Florence2Captioner(model_name="microsoft/Florence-2-base", detail_level=3)

# CaptionState: Singleton class to manage caption generation state
# CaptionState: 캡션 생성 상태를 관리하는 싱글톤 클래스
class CaptionState:
    _instance = None
    
    # Singleton pattern implementation to ensure only one instance exists
    # 단 하나의 인스턴스만 존재하도록 하는 싱글톤 패턴 구현
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CaptionState, cls).__new__(cls)
            cls._instance.current_caption = "Processing caption..."
            cls._instance.is_processing = False
            # Thread-safe lock for concurrent access
            # 동시 접근을 위한 스레드 안전 잠금장치
            cls._instance.lock = Lock()
        return cls._instance

    # Method to safely get current caption state
    # 현재 캡션 상태를 안전하게 가져오는 메서드
    def get_state(self):
        with self.lock:
            return {
                'caption': self.current_caption,
                'is_processing': self.is_processing
            }
    
    # Method to indicate caption processing has started
    # 캡션 처리가 시작되었음을 표시하는 메서드
    def start_processing(self):
        with self.lock:
            self.is_processing = True
            self.current_caption = "Processing caption..."
            print("Started processing caption")
    
    # Method to update state when caption generation is complete
    # 캡션 생성이 완료되었을 때 상태를 업데이트하는 메서드
    def finish_processing(self, caption):
        with self.lock:
            self.current_caption = caption
            self.is_processing = False
            print(f"Caption generated successfully: {caption[:50]}...")

# Create single instance of CaptionState
# CaptionState의 단일 인스턴스 생성
caption_state = CaptionState()

# Asynchronous function to generate image captions
# 이미지 캡션을 비동기적으로 생성하는 함수
def generate_caption_async(image_path):
    # Inner function that performs the actual caption generation
    # 실제 캡션 생성을 수행하는 내부 함수
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
    
    # Create and start a daemon thread for caption generation
    # 캡션 생성을 위한 데몬 스레드 생성 및 시작
    thread = Thread(target=_generate)
    # Thread will terminate when main program ends
    # 메인 프로그램 종료 시 스레드도 함께 종료됨
    thread.daemon = True
    thread.start()
    return thread