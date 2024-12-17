import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import whisper
import tempfile
import threading
import time
import cv2
from queue import Queue
from PIL import Image, ImageDraw, ImageFont

class WhisperSTTOverlay:
    def __init__(self, model_type="tiny"):
        self.model = whisper.load_model(model_type)
        self.sample_rate = 16000
        self.is_processing = False
        self.is_recording = False
        self.current_text = None
        self.text_timestamp = None
        self.text_display_duration = 10
        self.processing_queue = Queue()
        
        self.should_run = True
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()

    def _process_queue(self):
        while self.should_run:
            try:
                if not self.processing_queue.empty():
                    audio_data = self.processing_queue.get()
                    self._transcribe_audio(audio_data)
                time.sleep(0.1)
            except Exception as e:
                print(f"Queue processing error: {e}")

    def start_background_processing(self):
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_audio_queue(self):
        while True:
            try:
                if not self.processing_queue.empty():
                    audio_data = self.processing_queue.get()
                    self._transcribe_audio(audio_data)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in process_audio_queue: {e}")
                continue
    
    def start_recording(self, duration=5):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = threading.Thread(
                target=self._record_audio,
                args=(duration,)
            )
            self.recording_thread.daemon = True
            self.recording_thread.start()
            return True
        return False

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            return True
        return False

    def _record_audio(self, duration):
        try:
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            self.is_recording = False
            self.is_processing = True
            self.processing_queue.put(audio_data)
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            self.is_recording = False
            self.is_processing = False

    def _transcribe_audio(self, audio_data):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                wav.write(temp_audio_file.name, self.sample_rate, 
                         (audio_data * np.iinfo(np.int16).max).astype(np.int16))
                filename = temp_audio_file.name

            result = self.model.transcribe(
                filename,
                language="ko",
                task="transcribe",
                fp16=False
            )
            
            self.current_text = result["text"].strip()
            self.text_timestamp = time.time()
            
            os.remove(filename)
            
        except Exception as e:
            print(f"Transcription error: {e}")
            self.current_text = "음성 인식 실패"
            self.text_timestamp = time.time()
        finally:
            self.is_processing = False

    def get_text_position(self, frame_width, frame_height, overlay_bbox=None):
        text_y = frame_height - 50
        text_x = frame_width // 2
        
        if overlay_bbox:
            x, y, w, h = overlay_bbox
            overlay_center = x + w//2
            
            if x < frame_width // 2:
                text_x = int(frame_width * 0.75)
            else:
                text_x = int(frame_width * 0.25)
        
        return text_x, text_y

    def draw_text(self, frame):
        try:
            if frame is None:
                return

            if self.current_text and self.text_timestamp and time.time() - self.text_timestamp > self.text_display_duration:
                self.current_text = None
                self.text_timestamp = None
                return

            if self.is_recording:
                display_text = "Recording..."
            elif self.is_processing:
                display_text = "Processing..."
            elif self.current_text:
                display_text = self.current_text
            else:
                return

            try:
                frame_height, frame_width = frame.shape[:2]
                text_x = frame_width // 2
                text_y = frame_height - 50

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)

                try:
                    font = ImageFont.truetype("malgun.ttf", 30)  # Windows
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 30)  # Linux
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/nanum/NanumGothic.ttf", 30)
                        except:
                            print("Warning: Using default font as Korean fonts not found")
                            font = ImageFont.load_default()

                bbox = draw.textbbox((0, 0), display_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                bg_x1 = text_x - text_width//2 - 10
                bg_x2 = text_x + text_width//2 + 10
                bg_y1 = text_y - text_height - 10
                bg_y2 = text_y + 10

                overlay = frame.copy()
                cv2.rectangle(overlay,
                            (int(bg_x1), int(bg_y1)),
                            (int(bg_x2), int(bg_y2)),
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text(
                    (bg_x1 + 10, bg_y1 + 5),
                    display_text,
                    font=font,
                    fill=(255, 255, 255)
                )

                frame[:] = np.array(img_pil)

            except ImportError as e:
                print(f"PIL import error: {e}. Falling back to OpenCV text rendering")
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_w, text_h = text_size

                bg_x1 = text_x - text_w//2 - 10
                bg_x2 = text_x + text_w//2 + 10
                bg_y1 = text_y - text_h - 10
                bg_y2 = text_y + 10

                overlay = frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, display_text,
                        (text_x - text_w//2, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness)

        except Exception as e:
            print(f"Draw text error: {e}")
    
    def cleanup(self):
        self.should_run = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1)