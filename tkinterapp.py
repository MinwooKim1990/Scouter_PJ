import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import threading
from system import main as system_main

class ObjectTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Tracking System")
        
        # Create main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Create buttons frame
        self.create_control_panel()
        
        # Threading variables
        self.tracking_thread = None
        self.is_running = False

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.main_container, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Video selection
        ttk.Button(control_frame, text="Select Video", 
                  command=self.select_video).pack(side=tk.LEFT, padx=5)
        
        # Bing API Key entry
        ttk.Label(control_frame, text="Bing API Key:").pack(side=tk.LEFT, padx=5)
        self.bing_api_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.bing_api_var).pack(side=tk.LEFT, padx=5)
        
        # Save results checkbox
        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Save Results", 
                       variable=self.save_var).pack(side=tk.LEFT, padx=5)
        
        # Start/Stop buttons
        ttk.Button(control_frame, text="Start Tracking", 
                  command=self.start_tracking).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Tracking", 
                  command=self.stop_tracking).pack(side=tk.LEFT, padx=5)

    def select_video(self):
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if video_path:
            self.video_path = video_path

    def start_tracking(self):
        if hasattr(self, 'video_path') and not self.is_running:
            self.is_running = True
            # Start system_main in a separate thread
            self.tracking_thread = threading.Thread(
                target=self.run_tracking_system,
                args=(self.video_path, self.bing_api_var.get(), self.save_var.get())
            )
            self.tracking_thread.start()

    def run_tracking_system(self, video_path, bing_api, is_save):
        try:
            system_main(video_path=video_path, bing_api=bing_api, is_save=is_save)
        finally:
            self.is_running = False

    def stop_tracking(self):
        if self.is_running:
            self.is_running = False
            # OpenCV windows will be closed by system.py's cleanup

    def on_closing(self):
        self.stop_tracking()
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = ObjectTrackingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()