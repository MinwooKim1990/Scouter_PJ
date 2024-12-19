import os
import cv2
import queue
import threading
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk, filedialog
from tksystem import main as system_main

class VideoDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Tracking System")
        
        # Create queues for communication
        self.frame_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue()
        
        # To track the video source
        self.video_source = None
        
        # LLM 설정을 위한 변수들
        self.llm_provider = tk.StringVar(value="google")
        self.llm_model = tk.StringVar()
        self.llm_api_key = tk.StringVar()
        
        self.create_gui()

    def start_processing(self):
        """비디오 처리 시작"""
        self.system_thread = threading.Thread(
            target=system_main,
            args=(
                self.video_source, 
                self.bing_api_var.get(), 
                True, 
                self.frame_queue, 
                self.command_queue,
                self.llm_provider.get(),
                self.llm_api_key.get(),
                self.llm_model.get()
            )
        )
        self.system_thread.start()
        self.update_frame()
        
    def create_gui(self):
        # Control panel - 상단에 위치
        self.control_panel = ttk.Frame(self.root)
        self.control_panel.pack(fill=tk.X, pady=5, padx=5)
        
        # 왼쪽: 비디오 컨트롤
        video_controls = ttk.LabelFrame(self.control_panel, text="Video Controls")
        video_controls.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(video_controls, text="Open Video", 
                  command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_controls, text="Connect Cam", 
                  command=self.connect_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_controls, text="Play/Pause", 
                  command=self.toggle_pause).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시 라벨
        self.status_var = tk.StringVar(value="Status: No video source")
        ttk.Label(video_controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # 중앙: LLM 설정
        llm_frame = ttk.LabelFrame(self.control_panel, text="LLM Settings")
        llm_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # LLM Provider 선택
        ttk.Label(llm_frame, text="Provider:").pack(side=tk.LEFT, padx=5)
        providers = ["google", "openai", "groq"]
        provider_combo = ttk.Combobox(llm_frame, textvariable=self.llm_provider,
                                    values=providers, width=10, state="readonly")
        provider_combo.pack(side=tk.LEFT, padx=5)
        provider_combo.bind('<<ComboboxSelected>>', self.update_model_list)
        
        # LLM Model 선택
        ttk.Label(llm_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(llm_frame, textvariable=self.llm_model,
                                      width=20, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        # LLM API Key 입력
        ttk.Label(llm_frame, text="LLM API:").pack(side=tk.LEFT, padx=5)
        llm_api_entry = ttk.Entry(llm_frame, textvariable=self.llm_api_key, width=30)
        llm_api_entry.pack(side=tk.LEFT, padx=5)
        
        # 오른쪽: Bing API 설정
        bing_frame = ttk.LabelFrame(self.control_panel, text="Bing API Settings")
        bing_frame.pack(side=tk.RIGHT, padx=5, fill=tk.X)
        
        ttk.Label(bing_frame, text="Bing API:").pack(side=tk.LEFT, padx=5)
        self.bing_api_var = tk.StringVar()
        bing_api_entry = ttk.Entry(bing_frame, textvariable=self.bing_api_var, width=30)
        bing_api_entry.pack(side=tk.LEFT, padx=5)
        
        # 비디오 디스플레이 캔버스
        # 비디오 디스플레이 캔버스를 Frame으로 감싸기
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 마우스 이벤트 바인딩
        self.canvas.bind('<Button-1>', self.on_left_click)
        self.canvas.bind('<Button-3>', self.on_right_click)
        
        # bind_all 대신 focus_set()을 사용하여 캔버스에 포커스를 주고
        # 캔버스에 직접 키 이벤트를 바인딩
        self.canvas.focus_set()
        self.canvas.bind('<Tab>', self.on_tab_key)
        self.canvas.bind('<space>', self.on_space)
        self.canvas.bind('s', self.on_s_key)
        self.canvas.bind('t', self.on_t_key)
        self.canvas.bind('f', self.on_f_key)
        self.canvas.bind('a', self.on_a_key)
        
        # 버튼에 포커스가 가지 않도록 설정
        for child in video_controls.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(takefocus=False)
        
        # 초기 모델 리스트 업데이트
        self.update_model_list()

    def update_model_list(self, event=None):
        provider = self.llm_provider.get()
        models = {
            "google": ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
            "openai": ["gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18", "o1-2024-12-17", "o1-mini-2024-09-12", "gpt-3.5-turbo-0125"],
            "groq": ["llama-3.3-70b-versatile", "llama-3.2-90b-text-preview", "llama-3.2-11b-text-preview", "gemma2-9b-it", "mixtral-8x7b-32768",]
        }
        self.model_combo['values'] = models.get(provider, [])
        if models.get(provider):
            self.model_combo.set(models[provider][0])

    def toggle_pause(self):
        """재생/일시정지 토글"""
        if hasattr(self, 'system_thread') and self.system_thread.is_alive():
            self.command_queue.put({
                'type': 'space'
            })
        
    def open_video(self):
        """파일에서 비디오 열기"""
        if hasattr(self, 'system_thread') and self.system_thread.is_alive():
            self.command_queue.put({'type': 'stop'})
            self.system_thread.join()
            
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if video_path:
            # 비디오 크기 확인
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_dims = (width, height)
                cap.release()
                
                self.video_source = video_path
                self.start_processing()
                self.status_var.set(f"Status: Playing video - {os.path.basename(video_path)}")
    
    def on_left_click(self, event):
        if hasattr(self, 'system_thread'):
            # Get click coordinates relative to video frame
            click_point = self.get_video_coordinates(event.x, event.y)
            if click_point is not None:
                self.command_queue.put({
                    'type': 'click',
                    'point': click_point
                })

    def connect_camera(self):
        """웹캠 연결"""
        if hasattr(self, 'system_thread') and self.system_thread.is_alive():
            self.command_queue.put({'type': 'stop'})
            self.system_thread.join()
            
        # Test camera connection
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_dims = (width, height)
            cap.release()
            
            self.video_source = 0  # 웹캠 인덱스
            self.start_processing()
            self.status_var.set("Status: Camera connected")
        else:
            self.status_var.set("Status: Failed to connect camera")
            print("Cannot connect to camera")
    
    def on_right_click(self, event):
        self.command_queue.put({
            'type': 'right_click'
        })
    
    def on_space(self, event):
        event.widget.focus_set()  # 포커스 유지
        self.command_queue.put({
            'type': 'space'
        })
        return "break"
    
    def on_s_key(self, event):
        """Search mode toggle - Bing API key 확인"""
        if not self.bing_api_var.get().strip():
            print("Please enter Bing API key first")
            return
            
        self.command_queue.put({
            'type': 'key',
            'key': 's'
        })
        return "break"  # 이벤트 전파 중지
    
    def on_t_key(self, event):
        self.command_queue.put({
            'type': 'key',
            'key': 't'
        })
        return "break"  # 이벤트 전파 중지
    
    def on_f_key(self, event):
        self.command_queue.put({
            'type': 'key',
            'key': 'f'
        })
        return "break"  # 이벤트 전파 중지
    
    def on_a_key(self, event):
        self.command_queue.put({
            'type': 'key',
            'key': 'a'
        })
        return "break"  # 이벤트 전파 중지

    def on_tab_key(self, event):
        event.widget.focus_set()  # 포커스 유지
        self.command_queue.put({
            'type': 'key',
            'key': 'tab'
        })
        return "break"
    
    def get_video_coordinates(self, canvas_x, canvas_y):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if not hasattr(self, 'video_dims'):
            self.video_dims = (640, 480)  # 기본값
            
        actual_width, actual_height = self.video_dims
        
        # 캔버스에 표시된 이미지의 실제 크기와 위치 계산
        img_width = canvas_width
        img_height = int(img_width * (actual_height / actual_width))
        
        if img_height > canvas_height:
            img_height = canvas_height
            img_width = int(img_height * (actual_width / actual_height))
        
        # 이미지가 캔버스 중앙에 위치하도록 offset 계산
        x_offset = (canvas_width - img_width) // 2
        y_offset = (canvas_height - img_height) // 2
        
        # 클릭 좌표를 이미지 내부 좌표로 변환
        img_x = canvas_x - x_offset
        img_y = canvas_y - y_offset
        
        # 이미지 영역을 벗어난 클릭은 무시
        if img_x < 0 or img_x >= img_width or img_y < 0 or img_y >= img_height:
            return None
        
        # 실제 비디오 좌표로 변환
        video_x = int((img_x / img_width) * actual_width)
        video_y = int((img_y / img_height) * actual_height)
        
        return (video_x, video_y)
            
    def update_frame(self):
        try:
            if hasattr(self, 'system_thread') and self.frame_queue:
                try:
                    frame = self.frame_queue.get_nowait()
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    # Resize to fit canvas
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    if canvas_width > 1 and canvas_height > 1:  # Valid canvas size
                        img = self.resize_frame(img, canvas_width, canvas_height)
                        
                        # Calculate center position
                        x_offset = (canvas_width - img.width) // 2
                        y_offset = (canvas_height - img.height) // 2
                        
                        # Update canvas
                        self.photo = ImageTk.PhotoImage(image=img)
                        self.canvas.delete("all")
                        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
                except queue.Empty:
                    pass  # No new frame available
                    
        except Exception as e:
            print(f"Error updating frame: {e}")
        
        # Schedule next update
        self.root.after(10, self.update_frame)
            
    def resize_frame(self, img, width, height):
        if width <= 1 or height <= 1:  # Invalid dimensions
            return img
            
        # Calculate aspect ratio
        img_width, img_height = img.size
        aspect = img_width / img_height
        
        if width / height > aspect:
            new_height = height
            new_width = int(aspect * new_height)
        else:
            new_width = width
            new_height = int(new_width / aspect)
            
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
    def on_closing(self):
        """앱 종료 처리"""
        self.command_queue.put({'type': 'stop'})
        if hasattr(self, 'system_thread') and self.system_thread.is_alive():
            self.system_thread.join()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VideoDisplayApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.geometry("1280x720")
    root.mainloop()

if __name__ == "__main__":
    main()