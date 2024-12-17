import os
import cv2
import time
import torch
import numpy as np
from ultralytics import SAM
from utils.overlayed_display import ImageOverlay
from utils.whisper_overlay import WhisperSTTOverlay
from utils.srprocessor import SuperResolutionProcessor
from utils.caption import generate_caption_async, CaptionState

class ResourceManager:
    def __init__(self):
        self.click_point = None
        self.tracking_started = False
        self.initial_detection = True
        self.fixed_mode = True
        self.n = 1
        self.current_results = None
        
    def cleanup_resources(self):
        if self.current_results is not None:
            del self.current_results
            self.current_results = None
        torch.cuda.empty_cache()

def main(is_save = True):
    resource_manager = ResourceManager()
    IO = ImageOverlay()
    min_size = 256
    max_size = 512
    sr_processor = SuperResolutionProcessor(min_size, max_size)
    
    torch.cuda.empty_cache()
    
    with torch.cuda.device(0):
        model = SAM('models/mobile_sam.pt')
    
    video_path = 'data/squirrel.mp4'
    cap = cv2.VideoCapture(video_path)
    
    stt_overlay = WhisperSTTOverlay(model_type="tiny")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', IO.mouse_callback)
    
    fixed_sr_img = None
    last_cleanup_time = time.time()
    
    def cleanup_tracking():
        nonlocal fixed_sr_img
        resource_manager.cleanup_resources()
        if fixed_sr_img is not None:
            del fixed_sr_img
            fixed_sr_img = None
        IO.set_tracking_state(
            sr_img=None,
            should_show_overlay=False,
            click_point=None,
            tracking_started=False,
            initial_detection=True
        )
        torch.cuda.empty_cache()
    
    while cap.isOpened():
        try:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            tracking_state = IO.get_tracking_state()

            # 주기적인 메모리 정리 (10초마다)
            current_time = time.time()
            if current_time - last_cleanup_time > 10:
                torch.cuda.empty_cache()
                last_cleanup_time = current_time

            stt_overlay.draw_text(display_frame)

            if not tracking_state['tracking_started']:
                cv2.putText(display_frame, "Click left button to start tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Click right button to cancel tracking", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [t] key to start STT recording", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [t] key again to finish STT recording", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                if tracking_state['click_point'] is not None:
                    # 이전 결과 정리
                    if resource_manager.current_results is not None:
                        del resource_manager.current_results
                        
                    # 예측 실행 시 메모리 최적화
                    with torch.no_grad():
                        resource_manager.current_results = model.predict(
                            frame, 
                            points=[tracking_state['click_point']], 
                            labels=[1]
                        )
                    
                    results = resource_manager.current_results
                    
                    if len(results) > 0 and len(results[0].masks) > 0:
                        mask = results[0].masks.data[0].cpu().numpy()
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            bbox = cv2.boundingRect(largest_contour)
                            x, y, w, h = bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            should_process_sr = tracking_state['initial_detection'] or (not resource_manager.fixed_mode and tracking_state['prev_bbox'] != bbox)
                            
                            if should_process_sr:
                                cropped_img = IO.crop_object(frame, bbox, padding=20)
                                try:
                                    # 이전 SR 이미지 정리
                                    if fixed_sr_img is not None and not resource_manager.fixed_mode:
                                        del fixed_sr_img
                                    
                                    sr_img = sr_processor.process_image(cropped_img)
                                    IO.set_tracking_state(
                                        sr_img=sr_img,
                                        should_show_overlay=True,
                                        prev_bbox=bbox
                                    )
                                    
                                    if tracking_state['initial_detection']:
                                        panel = IO.create_info_panel(frame_height, frame_width)[0]
                                        panel_height, panel_width = panel.shape[:2]
                                        panel_x, panel_y = IO.calculate_panel_position(
                                            bbox=(x, y, w, h),
                                            panel_size=(panel_width, panel_height),
                                            frame_size=(frame_width, frame_height),
                                            margin=20
                                        )
                                        IO.set_tracking_state(
                                            panel_position=(panel_x, panel_y),
                                            initial_detection=False
                                        )
                                        
                                        if is_save:
                                            output_dir = 'result'
                                            os.makedirs(output_dir, exist_ok=True)
                                            output_path = os.path.join(output_dir, f'obj_{resource_manager.n}.jpg')
                                            sr_output_path = os.path.join(output_dir, f'obj_{resource_manager.n}_sr.jpg')
                                            frame_output_path = os.path.join(output_dir, f'frame_{resource_manager.n}.jpg')
                                            cv2.imwrite(output_path, cropped_img)
                                            cv2.imwrite(sr_output_path, sr_img)
                                            cv2.imwrite(frame_output_path, display_frame)
                                            
                                            caption_state = CaptionState()
                                            caption_state.start_processing()
                                            generate_caption_async(sr_output_path)
                                            
                                            fixed_sr_img = sr_img.copy()
                                            print(f"Saved: {output_path} and {sr_output_path}")
                                            resource_manager.n += 1
                                    elif not resource_manager.fixed_mode:
                                        fixed_sr_img = sr_img.copy()
                                    
                                except Exception as e:
                                    print(f"Error processing image: {e}")
                                    IO.set_tracking_state(
                                        sr_img=None,
                                        should_show_overlay=False
                                    )
                                    
                            if tracking_state['should_show_overlay'] and tracking_state['panel_position'] is not None:
                                display_frame = IO.overlay_image(
                                    display_frame,
                                    fixed_sr_img,
                                    tracking_state['panel_position'][0],
                                    tracking_state['panel_position'][1],
                                    original_bbox_size=(w, h),
                                    bbox=(x, y, w, h)
                                )
                            
                            IO.click_point = (x + w//2, y + h//2)
                
                cv2.putText(display_frame, "Right click to cancel tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Press 'f' to {'disable' if resource_manager.fixed_mode else 'enable'} fixed mode", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Mode: {'Fixed' if resource_manager.fixed_mode else 'Real-time'}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                current_fps = 1.0 / (time.time() - frame_start_time + 1e-6)
                cv2.putText(display_frame, f"FPS: {int(current_fps)}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in main loop: {e}")
            cleanup_tracking()
            break

        cv2.imshow('Tracking', display_frame)
        
        processing_time = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - processing_time) * 1000))
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27:
            break
        elif key == ord('f'):
            resource_manager.fixed_mode = not resource_manager.fixed_mode
            print(f"Switched to {'fixed' if resource_manager.fixed_mode else 'real-time'} mode")
        elif key == ord('t'):
            if not stt_overlay.is_recording:
                print("녹음을 시작합니다...")
                stt_overlay.start_recording(duration=5)
            else:
                print("녹음을 중지합니다.")
                stt_overlay.stop_recording()
        elif key == ord('c'):  # 'c' 키를 눌러 수동으로 메모리 정리
            cleanup_tracking()
    
    # 종료 시 정리
    cleanup_tracking()
    stt_overlay.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()