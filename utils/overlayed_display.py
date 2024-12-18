import cv2
import numpy as np
from utils.caption import CaptionState
from utils.search_state import SearchState

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
        self.search_results = None  # 검색 결과 저장
        self.search_panel_position = None  # 검색 패널 위치

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
            self.search_panel_position = None
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

        self.search_results = None
        self.search_panel_position = None

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
            'prev_bbox': self.prev_bbox,
            'search_results': self.search_results,
            'search_panel_position': self.search_panel_position
        }
    
    # Update tracking state with provided values
    # 제공된 값으로 추적 상태 업데이트
    def set_tracking_state(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_search_results(self, results):
        """검색 결과 설정"""
        self.search_results = results
    
    def create_search_panel(self, frame_height, frame_width):
        """검색 결과를 표시할 패널 생성"""
        # 메인 패널의 2/3 크기로 설정
        panel_height = int(frame_height * 5/6 * 1/2)  # 메인 패널(5/6)의 2/3 크기
        panel_width = int(frame_width * 5/22)  # 메인 패널과 동일한 너비
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        return panel, panel_height, panel_width
    
    def calculate_search_panel_position(self, bbox, panel_size, frame_size, margin=10):
        """검색 패널의 위치 계산"""
        x, y, w, h = bbox
        panel_width, panel_height = panel_size
        frame_width, frame_height = frame_size
        panel_gap = 5  # 패널 간 간격

        # bbox 중심점이 화면 중앙보다 왼쪽/오른쪽인지 확인
        bbox_center_x = x + w/2
        frame_center_x = frame_width/2

        # 패널 너비를 명시적으로 계산
        exact_panel_width = int(frame_width * 5/22)

        # x 위치 계산 - 패널 간격을 확실히 적용
        if bbox_center_x > frame_center_x:
            # bbox가 오른쪽에 있을 때
            main_panel_x = margin
            panel_x = main_panel_x + exact_panel_width + panel_gap  # 패널 간격 추가
        else:
            # bbox가 왼쪽에 있을 때
            main_panel_x = frame_width - exact_panel_width - margin
            panel_x = main_panel_x - exact_panel_width - panel_gap  # 패널 간격 추가

        # y 위치 계산
        panel_y = max(margin, (frame_height - frame_height * 5/6) // 2)
        
        return int(panel_x), int(panel_y)
    
    def render_search_results(self, panel, results):
        """검색 결과를 패널에 렌더링"""
        if not results:
            return panel

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_color = (200, 200, 200)
        line_spacing = 20
        left_margin = 15
        text_y = 20  # 시작 위치
        panel_height, panel_width = panel.shape[:2]
        max_text_width = panel_width - (left_margin * 2)

        # 상단 타이틀 추가
        title = "Bing Search Result"
        cv2.putText(panel, title, (left_margin, int(text_y)), font, 
                    font_scale, text_color, 1)  # text_y를 정수로 변환
        text_y += int(line_spacing * 1.5)  # 결과값을 정수로 변환

        # 검색 결과 섹션별로 렌더링
        sections = [
            ("Identification", [results.get("identified_object", "Unknown object")], False),
            ("Recent News", results.get("news", []), False),
            ("Web Results", results.get("web_results", []), False)
        ]

        for section_title, content, is_bold in sections:
            cv2.putText(panel, section_title, (left_margin, int(text_y)), font,
                    font_scale, text_color, 1 if is_bold else 1)
            text_y += line_spacing

            for text in content:
                wrapped_lines = self.wrap_text(text, font, font_scale, max_text_width)
                for line in wrapped_lines:
                    cv2.putText(panel, line, (left_margin, int(text_y)), font,
                            font_scale, text_color, 1)
                    text_y += line_spacing
            text_y += 5  # 섹션 간 간격

        return panel

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
        panel_width = int(frame_width * 5/22)
        
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        img_height = int(panel_height * 1/3) 
        cv2.line(panel, (0, img_height), (panel_width, img_height), (100, 100, 100), 1)
        
        return panel, img_height

    # Calculate optimal panel position based on bbox and frame size
    # 경계 상자와 프레임 크기를 기반으로 최적의 패널 위치 계산
    def calculate_panel_position(self, bbox, panel_size, frame_size, margin=10):
        """메인 패널 위치 계산"""
        x, y, w, h = bbox
        panel_width, panel_height = panel_size
        frame_width, frame_height = frame_size

        # bbox 중심점이 화면 중앙보다 왼쪽/오른쪽인지 확인
        bbox_center_x = x + w/2
        frame_center_x = frame_width/2

        if bbox_center_x > frame_center_x:
            # bbox가 오른쪽에 있으면 패널은 왼쪽 끝에 위치
            panel_x = margin
        else:
            # bbox가 왼쪽에 있으면 패널은 오른쪽 끝에 위치
            panel_x = frame_width - panel_width - margin
        
        # y 위치는 동일하게 유지
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
            # Calculate maximum range of image display
            # 이미지를 표시할 수 있는 최대 영역 계산
            max_img_width = panel_width - 10
            max_img_height = img_height - 10
            
            # Calculate original image ratio
            # 원본 이미지의 비율 계산
            orig_h, orig_w = overlay.shape[:2]
            aspect_ratio = orig_w / orig_h
            
            # Calculate image size
            # 이미지 크기 계산
            if aspect_ratio > 1:  # Width longer image / 가로가 더 긴 이미지
                display_w = max_img_width
                display_h = int(display_w / aspect_ratio)
                if display_h > max_img_height:  # If the height over the maximum hight, recalculate / 높이가 초과하면 높이 기준으로 다시 계산
                    display_h = max_img_height
                    display_w = int(display_h * aspect_ratio)
            else:  # Height longer image / 세로가 더 긴 이미지
                display_h = max_img_height
                display_w = int(display_h * aspect_ratio)
                if display_w > max_img_width:  # If the width over the maximum hight, recalculate / 너비가 초과하면 너비 기준으로 다시 계산
                    display_w = max_img_width
                    display_h = int(display_w / aspect_ratio)
            
            # Image resize
            # 이미지 리사이즈
            display_img = cv2.resize(overlay, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
            
            # Offset calculation for locating image in the center
            # 이미지를 패널의 중앙에 배치하기 위한 오프셋 계산
            x_offset = (panel_width - display_w) // 2
            y_offset = (img_height - display_h) // 2
            
            # Locate image to panel
            # 이미지를 패널에 배치
            h, w = display_img.shape[:2]
            panel[y_offset:y_offset+h, x_offset:x_offset+w] = display_img
            
            # Configure text display settings
            # 텍스트 표시 설정 구성
            # 텍스트 표시 설정
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_color = (200, 200, 200)
            line_spacing = 25
            left_margin = 15
            
            # "Upscaled Zoom In image" 텍스트를 이미지 라인 바로 위에 중앙 정렬
            zoom_text = "[Upscaled Zoom In Image]"
            text_size = cv2.getTextSize(zoom_text, font, font_scale, 1)[0]
            text_x = (panel_width - text_size[0]) // 2  # 중앙 정렬을 위한 x 좌표
            text_y = img_height - 10  # 라인에 더 가깝게 배치
            
            # 볼드 효과를 위한 텍스트 두 번 그리기
            cv2.putText(panel, zoom_text, (text_x+1, text_y), font, font_scale, text_color, 1)
            cv2.putText(panel, zoom_text, (text_x, text_y), font, font_scale, text_color, 1)
            
            max_text_width = panel_width - (left_margin * 2)
            
            # 캡션 상태 가져오기
            state = CaptionState().get_state()
            caption = state['caption']
            is_processing = state['is_processing']
            
            # 나머지 텍스트 블록 (Description 등) 렌더링
            text_y = img_height + 30
            text_blocks = [
                ["Description", True],
                [caption, False]
            ]
            
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
                    
            # 검색 패널 관련 코드
            search_state = SearchState()
            state = search_state.get_state()
            if state['results'] or state['is_processing']:
                search_panel, _, _ = self.create_search_panel(bg_h, bg_w)
                
                if state['is_processing']:
                    cv2.putText(search_panel, "Searching...", (15, 30), font, 
                            font_scale, text_color, 1)
                else:
                    search_panel = self.render_search_results(search_panel, state['results'])
                
                # 검색 패널 위치 계산 (처음 한 번만)
                if self.search_panel_position is None:
                    self.search_panel_position = self.calculate_search_panel_position(
                        bbox=bbox,
                        panel_size=(search_panel.shape[1], search_panel.shape[0]),
                        frame_size=(bg_w, bg_h)
                    )

                # 검색 패널 블렌딩
                if self.search_panel_position is not None:
                    search_panel_x, search_panel_y = self.search_panel_position
                    search_panel_h, search_panel_w = search_panel.shape[:2]
                    
                    # 좌표와 크기를 정수로 변환
                    search_panel_x = int(search_panel_x)
                    search_panel_y = int(search_panel_y)
                    search_panel_h = int(search_panel_h)
                    search_panel_w = int(search_panel_w)
                    
                    # 검색 패널이 화면 안에 있는지 확인 후 블렌딩
                    if (search_panel_y + search_panel_h <= bg_h and 
                        search_panel_x + search_panel_w <= bg_w and
                        search_panel_y >= 0 and
                        search_panel_x >= 0):
                        alpha = 0.05
                        roi = background[search_panel_y:search_panel_y+search_panel_h, 
                                    search_panel_x:search_panel_x+search_panel_w]
                        if roi.shape == search_panel.shape:
                            background[search_panel_y:search_panel_y+search_panel_h, 
                                    search_panel_x:search_panel_x+search_panel_w] = \
                                cv2.addWeighted(roi, alpha, search_panel, 1 - alpha, 0)

            # 메인 패널 블렌딩
            alpha = 0.05
            try:
                panel_h, panel_w = panel.shape[:2]
                if (panel_y + panel_h <= bg_h and 
                    panel_x + panel_w <= bg_w and
                    panel_y >= 0 and
                    panel_x >= 0):
                    roi = background[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w]
                    background[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = \
                        cv2.addWeighted(roi, alpha, panel, 1 - alpha, 0)
            except ValueError as e:
                print(f"Panel positioning error: {e}")
                
        except Exception as e:
            print(f"Overlay error: {e}")
        
        return background