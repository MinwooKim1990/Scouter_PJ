import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import queue
import threading
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
        
        self.create_gui()

    def start_processing(self):
        """비디오 처리 시작"""
        self.system_thread = threading.Thread(
            target=system_main,
            args=(self.video_source, self.bing_api_var.get(), True, self.frame_queue, self.command_queue)
        )
        self.system_thread.start()
        self.update_frame()
        
    def create_gui(self):
        # Control panel
        self.control_panel = ttk.Frame(self.root)
        self.control_panel.pack(fill=tk.X, pady=5, padx=5)
        
        # Video controls
        video_controls = ttk.Frame(self.control_panel)
        video_controls.pack(side=tk.LEFT, padx=5)
        
        # Video source buttons
        ttk.Button(video_controls, text="Open Video", 
                  command=self.open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_controls, text="Connect Cam", 
                  command=self.connect_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_controls, text="Play/Pause", 
                  command=self.toggle_pause).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Status: No video source")
        ttk.Label(video_controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)
        
        # Bing API Key input
        api_frame = ttk.LabelFrame(self.control_panel, text="API Settings")
        api_frame.pack(side=tk.RIGHT, padx=5, fill=tk.X)
        
        ttk.Label(api_frame, text="Bing API Key:").pack(side=tk.LEFT, padx=5)
        self.bing_api_var = tk.StringVar()
        ttk.Entry(api_frame, textvariable=self.bing_api_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # Video display canvas
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_left_click)
        self.canvas.bind('<Button-3>', self.on_right_click)
        
        # Bind keyboard events
        self.root.bind('s', self.on_s_key)
        self.root.bind('t', self.on_t_key)
        self.root.bind('f', self.on_f_key)

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
        self.command_queue.put({
            'type': 'space'
        })
    
    def on_s_key(self, event):
        """Search mode toggle - Bing API key 확인"""
        if not self.bing_api_var.get().strip():
            print("Please enter Bing API key first")
            return
            
        self.command_queue.put({
            'type': 'key',
            'key': 's'
        })
    
    def on_t_key(self, event):
        self.command_queue.put({
            'type': 'key',
            'key': 't'
        })
    
    def on_f_key(self, event):
        self.command_queue.put({
            'type': 'key',
            'key': 'f'
        })
    
    def get_video_coordinates(self, canvas_x, canvas_y):
        """실제 비디오 좌표로 변환"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if not hasattr(self, 'video_dims'):
            # 실제 비디오 크기를 저장할 속성 추가
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
        """프레임 업데이트 함수"""
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
                        
                        # Update canvas
                        self.photo = ImageTk.PhotoImage(image=img)
                        self.canvas.delete("all")
                        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
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