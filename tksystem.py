import os
import cv2
import time
import queue
import torch
import argparse
import numpy as np
from ultralytics import SAM
from utils.overlayed_display import ImageOverlay
from utils.whisper_overlay import WhisperSTTOverlay
from utils.bing_search_api import BingSearchProcessor
from utils.srprocessor import SuperResolutionProcessor
from utils.search_state import SearchState, search_async
from utils.caption import generate_caption_async, CaptionState

# ResourceManager: Class to manage system resources and tracking states
# 시스템 자원과 추적 상태를 관리하는 클래스
class ResourceManager:
    def __init__(self, bing_api=None):
        # Store mouse click coordinates / 마우스 클릭 좌표 저장
        self.click_point = None  
        # Flag for tracking status / 추적 시작 여부 플래그
        self.tracking_started = False  
        # Flag for first detection / 첫 감지 여부 플래그
        self.initial_detection = True  
        # Toggle between fixed and real-time mode / 고정 모드와 실시간 모드 전환
        self.fixed_mode = True  
        # Counter for saved images / 저장된 이미지 카운터
        self.n = 1  
        # Store current detection results / 현재 감지 결과 저장
        self.current_results = None  
        # Search mode flag / 검색 모드 플래그 추가
        self.search_mode = False  
        # Initialize Bing search processor / Bing 검색 프로세서 초기화
        self.bing_processor = BingSearchProcessor(bing_api)
        # Initialize search state / 검색 상태 초기화
        self.search_state = SearchState(bing_api)
        # Pause state tracking / 일시정지 상태를 추적하는 변수
        self.is_paused = False  
        # Last processed point to avoid redundant processing / 중복 처리를 방지하기 위한 마지막 처리 지점
        self.last_processed_point = None
        
    # Clean up GPU memory and reset results
    # GPU 메모리 정리 및 결과 초기화
    def cleanup_resources(self):
        """
        Clean up GPU memory and reset results
        GPU 메모리를 정리하고 결과를 초기화하는 함수
        """
        if self.current_results is not None:
            del self.current_results
            self.current_results = None
        torch.cuda.empty_cache()

    def toggle_search_mode(self):
        """
        Toggle search mode on/off
        검색 모드를 켜고 끄는 토글 함수
        """
        self.search_mode = not self.search_mode
        self.search_state.set_search_mode(self.search_mode)
        return self.search_mode
    
    def toggle_pause(self):
        """
        Toggle video pause state
        비디오 일시정지 상태를 토글하는 함수
        """
        self.is_paused = not self.is_paused
        return self.is_paused
    
    def should_process_frame(self, current_point):
        """
        Determine if the current frame should be processed
        현재 프레임을 처리해야 하는지 결정하는 함수
        
        Args:
            current_point: Current mouse click position / 현재 마우스 클릭 위치
        Returns:
            bool: Whether to process the frame / 프레임 처리 여부
        """
        if not self.is_paused:
            return True
        if current_point != self.last_processed_point:
            self.last_processed_point = current_point
            return True
        return False

def main(video_path='data/squirrel.mp4', bing_api=None, is_save=True, frame_queue=None, command_queue=None):
    """
    수정된 main 함수. 프레임 큐와 명령 큐를 통해 GUI와 통신
    
    Args:
        video_path: Path to input video / 입력 비디오 경로
        bing_api: Bing API key for search functionality / 검색 기능을 위한 Bing API 키
        is_save: Flag to save processed images / 처리된 이미지 저장 여부
        frame_queue: Queue for sending frames to GUI / GUI로 프레임을 전송하기 위한 큐
        command_queue: Queue for receiving commands from GUI / GUI로부터 명령을 받기 위한 큐
    """
    resource_manager = ResourceManager(bing_api=bing_api)
    IO = ImageOverlay()
    min_size = 256
    max_size = 512
    sr_processor = SuperResolutionProcessor(min_size, max_size)
    torch.cuda.empty_cache()
    
    # Initialize model
    with torch.cuda.device(0):
        model = SAM('models/mobile_sam.pt')

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    stt_overlay = WhisperSTTOverlay(model_type="tiny")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    # Initialize variables
    last_cleanup_time = time.time()
    current_frame = None
    fixed_sr_img = None
    
    def cleanup_tracking():
        """Clean up resources and reset tracking state"""
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
            initial_detection=True,
            search_results=None,
            search_panel_position=None
        )
        torch.cuda.empty_cache()

    # Main processing loop
    while cap.isOpened():
        try:
            # Process commands from GUI
            if command_queue and not command_queue.empty():
                cmd = command_queue.get()
                if cmd.get('type') == 'click':
                    click_point = cmd['point']
                    IO.click_point = click_point
                    IO.tracking_started = True
                    # tracking_state도 직접 업데이트
                    IO.set_tracking_state(
                        click_point=click_point,
                        tracking_started=True,
                        initial_detection=True
                    )
                elif cmd.get('type') == 'right_click':
                    cleanup_tracking()
                elif cmd.get('type') == 'space':
                    resource_manager.toggle_pause()
                elif cmd.get('type') == 'key':
                    if cmd['key'] == 's':
                        search_mode = resource_manager.toggle_search_mode()
                        print(f"Search mode {'enabled' if search_mode else 'disabled'}")
                    elif cmd['key'] == 't':
                        if not stt_overlay.is_recording:
                            print("녹음을 시작합니다...")
                            stt_overlay.start_recording(duration=5)
                        else:
                            print("녹음을 중지합니다.")
                            stt_overlay.stop_recording()
                    elif cmd['key'] == 'f':
                        resource_manager.fixed_mode = not resource_manager.fixed_mode
                        print(f"Switched to {'fixed' if resource_manager.fixed_mode else 'real-time'} mode")
                elif cmd.get('type') == 'stop':
                    break

            frame_start_time = time.time()
            
            # Read frame
            if not resource_manager.is_paused:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                current_frame = frame.copy()
            else:
                if current_frame is None:
                    continue
                frame = current_frame.copy()
                
            display_frame = frame.copy()
            tracking_state = IO.get_tracking_state()

            # Periodic cleanup
            current_time = time.time()
            if current_time - last_cleanup_time > 10:
                torch.cuda.empty_cache()
                last_cleanup_time = current_time

            # Draw STT overlay
            stt_overlay.draw_text(display_frame)

            # Display instructions or process tracking
            if not tracking_state['tracking_started']:
                cv2.putText(display_frame, "Click left button to start tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Click right button to cancel tracking", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [t] key to start STT recording", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [t] key again to finish STT recording", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [s] key to toggle search mode", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Push [Space] to pause/resume video", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if tracking_state['tracking_started']:
                if tracking_state['click_point'] is not None:
                    current_point = tracking_state['click_point']
                    if resource_manager.should_process_frame(current_point):
                        if resource_manager.current_results is not None:
                            del resource_manager.current_results
                        
                        # Perform object detection
                        with torch.no_grad():
                            resource_manager.current_results = model.predict(
                                frame, 
                                points=[tracking_state['click_point']], 
                                labels=[1]
                            )
                        
                    if resource_manager.current_results is not None:
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

                                                if resource_manager.search_mode:
                                                    search_async(sr_output_path, resource_manager.search_state)
                                                
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
                
                # Display status information
                cv2.putText(display_frame, "Right click to cancel tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Press 'f' to {'disable' if resource_manager.fixed_mode else 'enable'} fixed mode", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Mode: {'Fixed' if resource_manager.fixed_mode else 'Real-time'}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Search Mode: {'On' if resource_manager.search_mode else 'Off'}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Video: {'Paused' if resource_manager.is_paused else 'Playing'}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                current_fps = 1.0 / (time.time() - frame_start_time + 1e-6)
                cv2.putText(display_frame, f"Processing Speed: {int(current_fps)}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Send frame to GUI
            if frame_queue is not None:
                try:
                    frame_queue.put_nowait(display_frame)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()  # Remove oldest frame
                        frame_queue.put_nowait(display_frame)
                    except:
                        pass

            # Control frame rate
            processing_time = time.time() - frame_start_time
            if processing_time < frame_delay:
                time.sleep(frame_delay - processing_time)

        except Exception as e:
            print(f"Error in main loop: {e}")
            break

    # Cleanup
    cap.release()
    if fixed_sr_img is not None:
        del fixed_sr_img
    torch.cuda.empty_cache()
