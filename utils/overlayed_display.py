import cv2
import numpy as np
from utils.caption import CaptionState

class ImageOverlay:
    # Initialize the overlay manager with default states
    # 오버레이 관리자를 기본 상태로 초기화
    def __init__(self):
        # Stores mouse click coordinates
        # 마우스 클릭 좌표 저장
        self.click_point = None
        # Flag for object tracking status
        # 객체 추적 상태 플래그
        self.tracking_started = False
        # Flag for first detection
        # 첫 감지 플래그
        self.initial_detection = False
        # Super-resolution image storage
        # 고해상도 이미지 저장
        self.sr_img = None
        # Control overlay visibility
        # 오버레이 표시 여부 제어
        self.should_show_overlay = False
        # Information panel position
        # 정보 패널 위치
        self.panel_position = None
        # Previous bounding box
        # 이전 경계 상자
        self.prev_bbox = None

    # Mouse event handler for tracking control
    # 추적 제어를 위한 마우스 이벤트 핸들러
    def mouse_callback(self, event, x, y, flags, param):
        # Left click starts tracking
        # 왼쪽 클릭으로 추적 시작
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            self.tracking_started = True
            self.initial_detection = True
            self.sr_img = None
            self.should_show_overlay = False
            self.panel_position = None
            self.prev_bbox = None
        # Right click resets tracking
        # 오른쪽 클릭으로 추적 초기화
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_tracking_state()

    # Reset all tracking-related states to default
    # 모든 추적 관련 상태를 기본값으로 재설정
    def reset_tracking_state(self):
        self.click_point = None
        self.tracking_started = False
        self.initial_detection = False
        self.sr_img = None
        self.should_show_overlay = False
        self.panel_position = None
        self.prev_bbox = None

    # Get current tracking state as dictionary
    # 현재 추적 상태를 딕셔너리로 반환
    def get_tracking_state(self):
        return {
            'click_point': self.click_point,
            'tracking_started': self.tracking_started,
            'initial_detection': self.initial_detection,
            'sr_img': self.sr_img,
            'should_show_overlay': self.should_show_overlay,
            'panel_position': self.panel_position,
            'prev_bbox': self.prev_bbox
        }
    
    # Update tracking state with provided values
    # 제공된 값으로 추적 상태 업데이트
    def set_tracking_state(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # Crop object from frame using bounding box with padding
    # 경계 상자와 패딩을 사용하여 프레임에서 객체 잘라내기
    def crop_object(self, frame, bbox, padding = None):
        x, y, w, h = bbox
        x, y, w, h = x-padding, y-padding, w+(2*padding), h+(2*padding)
        return frame[y:y+h, x:x+w]

    # Calculate display size while maintaining aspect ratio
    # 종횡비를 유지하면서 표시 크기 계산
    def calculate_display_size(self, original_bbox_size, sr_img_size, available_space, min_scale=1.5, max_scale=3.0):
        orig_w, orig_h = original_bbox_size
        avail_w, avail_h = available_space

        scale = min(
            max_scale, 
            max(
                min_scale, 
                avail_w / orig_w, 
                avail_h / orig_h  
            )
        )
        
        display_w = min(int(orig_w * scale), avail_w)
        display_h = min(int(orig_h * scale), avail_h)
        
        return display_w, display_h

    # Clean text by removing extra whitespace
    # 추가 공백을 제거하여 텍스트 정리
    def clean_text(self, text):
        return ' '.join(text.split())

    # Wrap text to fit within specified width
    # 지정된 너비에 맞게 텍스트 줄바꿈
    def wrap_text(self, text, font, font_scale, max_width):
        text = self.clean_text(text)
        words = text.split(' ')
        wrapped_lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_width = cv2.getTextSize(' '.join(current_line), font, font_scale, 1)[0][0]
            
            if line_width > max_width:
                if len(current_line) > 1:
                    current_line.pop()
                    wrapped_lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    wrapped_lines.append(word)
                    current_line = []

        if current_line:
            wrapped_lines.append(' '.join(current_line))
        
        return wrapped_lines

    # Create information panel with specified dimensions
    # 지정된 크기로 정보 패널 생성
    def create_info_panel(self, frame_height, frame_width):
        panel_height = int(frame_height * 5/6)
        panel_width = int(frame_width * 3/10)
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        img_height = int(panel_height * 2/5) 
        cv2.line(panel, (0, img_height), (panel_width, img_height), (100, 100, 100), 1)
        
        return panel, img_height

    # Calculate optimal panel position based on bbox and frame size
    # 경계 상자와 프레임 크기를 기반으로 최적의 패널 위치 계산
    def calculate_panel_position(self, bbox, panel_size, frame_size, margin=20):
        x, y, w, h = bbox
        panel_width, panel_height = panel_size
        frame_width, frame_height = frame_size
        
        panel_x = x + w + margin
        
        if panel_x + panel_width > frame_width - margin:
            if x > panel_width + margin:
                panel_x = x - panel_width - margin
            else:
                panel_x = frame_width - panel_width - margin
        
        panel_y = max(margin, (frame_height - panel_height) // 2)
        
        return panel_x, panel_y

    # Overlay image and information panel on background
    # 배경에 이미지와 정보 패널 오버레이
    def overlay_image(self, background, overlay, panel_x, panel_y, original_bbox_size, bbox):
        if overlay is None or background is None:
            return background
            
        bg_h, bg_w = background.shape[:2]
        panel, img_height = self.create_info_panel(bg_h, bg_w)
        panel_height, panel_width = panel.shape[:2]
        
        try:
            # Prepare and position the display image
            # 표시 이미지 준비 및 위치 지정
            img_width = panel_width - 20 
            display_img = cv2.resize(overlay, (img_width, img_height-20), interpolation=cv2.INTER_LINEAR)
            
            y_offset = 10
            x_offset = 10
            h, w = display_img.shape[:2]
            panel[y_offset:y_offset+h, x_offset:x_offset+w] = display_img
            
            # Configure text display settings
            # 텍스트 표시 설정 구성
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_color = (200, 200, 200)
            line_spacing = 25
            left_margin = 15
            
            max_text_width = panel_width - (left_margin * 2)
            
            # Get caption state and prepare text blocks
            # 캡션 상태 가져오기 및 텍스트 블록 준비
            state = CaptionState().get_state()
            caption = state['caption']
            is_processing = state['is_processing']
            
            print(f"Current caption state - Processing: {is_processing}, Caption: {caption[:50]}...")
            
            text_blocks = [
                [" ***Upscaled Zoom In image***", True],
                ["Description", True],
                [caption, False]
            ]
            
            # Render text blocks with proper formatting
            # 적절한 형식으로 텍스트 블록 렌더링
            text_y = img_height + 30
            
            for text, is_bold in text_blocks:
                if text:
                    wrapped_lines = self.wrap_text(text, font, font_scale, max_text_width)
                    for line in wrapped_lines:
                        if is_bold:
                            cv2.putText(panel, line, (left_margin+1, text_y), font, font_scale, text_color, 1)
                        cv2.putText(panel, line, (left_margin, text_y), font, font_scale, text_color, 1)
                        text_y += line_spacing
                else:
                    text_y += line_spacing
            
            # Blend panel with background
            # 패널을 배경과 블렌딩
            alpha = 0.05
            try:
                background[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width] = \
                    cv2.addWeighted(
                        background[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width],
                        alpha,
                        panel,
                        1 - alpha,
                        0
                    )
            except ValueError as e:
                print(f"Panel positioning error: {e}")
            
        except Exception as e:
            print(f"Overlay error: {e}")
        
        return background