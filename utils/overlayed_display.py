from utils.caption import CaptionState
import cv2
import numpy as np

class ImageOverlay:
    def __init__(self):
        self.click_point = None
        self.tracking_started = False
        self.initial_detection = False
        self.sr_img = None
        self.should_show_overlay = False
        self.panel_position = None
        self.prev_bbox = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            self.tracking_started = True
            self.initial_detection = True
            self.sr_img = None
            self.should_show_overlay = False
            self.panel_position = None
            self.prev_bbox = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_tracking_state()
    def reset_tracking_state(self):
        self.click_point = None
        self.tracking_started = False
        self.initial_detection = False
        self.sr_img = None
        self.should_show_overlay = False
        self.panel_position = None
        self.prev_bbox = None

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
    
    def set_tracking_state(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def crop_object(self, frame, bbox, padding = None):
        x, y, w, h = bbox
        x, y, w, h = x-padding, y-padding, w+(2*padding), h+(2*padding)
        return frame[y:y+h, x:x+w]

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

    def clean_text(self, text):
        return ' '.join(text.split())

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
    def create_info_panel(self, frame_height, frame_width):

        panel_height = int(frame_height * 5/6)
        panel_width = int(frame_width * 3/10)
        

        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        img_height = int(panel_height * 2/5) 
        cv2.line(panel, (0, img_height), (panel_width, img_height), (100, 100, 100), 1)
        
        return panel, img_height

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

    def overlay_image(self, background, overlay, panel_x, panel_y, original_bbox_size, bbox):
        if overlay is None or background is None:
            return background
            
        bg_h, bg_w = background.shape[:2]
        panel, img_height = self.create_info_panel(bg_h, bg_w)
        panel_height, panel_width = panel.shape[:2]
        
        try:
            img_width = panel_width - 20 
            display_img = cv2.resize(overlay, (img_width, img_height-20), interpolation=cv2.INTER_LINEAR)
            
            y_offset = 10
            x_offset = 10
            h, w = display_img.shape[:2]
            panel[y_offset:y_offset+h, x_offset:x_offset+w] = display_img
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            text_color = (200, 200, 200)
            line_spacing = 25
            left_margin = 15
            
            max_text_width = panel_width - (left_margin * 2)
            
            state = CaptionState().get_state()
            caption = state['caption']
            is_processing = state['is_processing']
            
            print(f"Current caption state - Processing: {is_processing}, Caption: {caption[:50]}...")
            
            text_blocks = [
                [" ***Upscaled Zoom In image***", True],
                ["Description", True],
                [caption, False]
            ]
            
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