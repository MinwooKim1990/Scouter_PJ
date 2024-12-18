import os
import cv2
import time
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

def main(video_path='data/squirrel.mp4', bing_api=None, is_save=True):
    """
    Main function to run the object tracking system
    객체 추적 시스템을 실행하는 메인 함수
    
    Args:
        video_path: Path to input video / 입력 비디오 경로
        bing_api: Bing API key for search functionality / 검색 기능을 위한 Bing API 키
        is_save: Flag to save processed images / 처리된 이미지 저장 여부
    """
    # Initialize core components
    # 핵심 구성 요소 초기화
    resource_manager = ResourceManager(bing_api=bing_api)
    IO = ImageOverlay()
    # Minimum size for super resolution
    # 초해상도의 최소 크기
    min_size = 256
    # Maximum size for super resolution
    # 초해상도의 최대 크기
    max_size = 512
    sr_processor = SuperResolutionProcessor(min_size, max_size)
    torch.cuda.empty_cache()
    
    # Load SAM (Segment Anything Model)
    # SAM 모델 로드
    with torch.cuda.device(0):
        model = SAM('models/mobile_sam.pt')

    # Set up video capture and STT overlay
    # 비디오 캡처 및 음성인식 오버레이 설정
    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    stt_overlay = WhisperSTTOverlay(model_type="tiny")
    
    # Get video properties
    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Calculate delay between frames
    # 프레임 간 지연 시간 계산
    frame_delay = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up display window
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', IO.mouse_callback)
    
    fixed_sr_img = None
    last_cleanup_time = time.time()
    current_frame = None
    
    def cleanup_tracking():
        """
        Clean up resources and reset tracking state
        리소스를 정리하고 추적 상태를 초기화하는 함수
        """
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
    # 메인 처리 루프
    while cap.isOpened():
        try:
            # Record start time for FPS calculation
            # FPS 계산을 위한 시작 시간 기록
            frame_start_time = time.time()
            
            # Read frame based on pause state
            # 일시정지 상태에 따라 프레임 읽기
            if not resource_manager.is_paused:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame = frame.copy()
            else:
                frame = current_frame.copy()
                
            # Create copy for display
            # 화면 표시를 위한 프레임 복사
            display_frame = frame.copy()
            tracking_state = IO.get_tracking_state()

            # Periodic memory cleanup
            # 주기적인 메모리 정리
            current_time = time.time()
            if current_time - last_cleanup_time > 10:
                torch.cuda.empty_cache()
                last_cleanup_time = current_time

            # Draw STT (Speech-to-Text) results
            # 음성인식 결과 표시
            stt_overlay.draw_text(display_frame)

            # Display instructions when not tracking
            # 추적 중이 아닐 때 사용 설명서 표시
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
            else:
                # Process tracking when active
                # 추적이 활성화되었을 때 처리
                if tracking_state['click_point'] is not None:
                    # Check if frame should be processed
                    # 프레임 처리 여부 확인
                    if resource_manager.should_process_frame(tracking_state['click_point']):
                        if resource_manager.current_results is not None:
                            del resource_manager.current_results
                        
                        # Perform object detection using SAM
                        # SAM을 사용하여 객체 감지 수행
                        with torch.no_grad():
                            resource_manager.current_results = model.predict(
                                frame, 
                                points=[tracking_state['click_point']], 
                                labels=[1]
                            )
                        
                    # Process detection results
                    # 감지 결과 처리
                    if resource_manager.current_results is not None:
                        results = resource_manager.current_results
                        if len(results) > 0 and len(results[0].masks) > 0:
                            # Extract mask and find contours
                            # 마스크 추출 및 윤곽선 찾기
                            mask = results[0].masks.data[0].cpu().numpy()
                            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                        cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                # Find largest contour and its bounding box
                                # 가장 큰 윤곽선과 경계 상자 찾기
                                largest_contour = max(contours, key=cv2.contourArea)
                                bbox = cv2.boundingRect(largest_contour)
                                x, y, w, h = bbox
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                
                                # Determine if super-resolution should be applied
                                # 초해상도 적용 여부 결정
                                should_process_sr = tracking_state['initial_detection'] or (not resource_manager.fixed_mode and tracking_state['prev_bbox'] != bbox)
                                
                                if should_process_sr:
                                    # Crop and process object image
                                    # 객체 이미지 자르기 및 처리
                                    cropped_img = IO.crop_object(frame, bbox, padding=20)
                                    try:
                                        # Clean up previous super-resolution image
                                        # 이전 초해상도 이미지 정리
                                        if fixed_sr_img is not None and not resource_manager.fixed_mode:
                                            del fixed_sr_img
                                        
                                        # Apply super-resolution
                                        # 초해상도 적용
                                        sr_img = sr_processor.process_image(cropped_img)
                                        IO.set_tracking_state(
                                            sr_img=sr_img,
                                            should_show_overlay=True,
                                            prev_bbox=bbox
                                        )

                                        # Handle initial detection
                                        # 초기 감지 처리
                                        if tracking_state['initial_detection']:
                                            # Create information panel
                                            # 정보 패널 생성
                                            panel = IO.create_info_panel(frame_height, frame_width)[0]
                                            panel_height, panel_width = panel.shape[:2]
                                            # Calculate panel position
                                            # 패널 위치 계산
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

                                            # Save results if enabled
                                            # 결과 저장 (활성화된 경우)
                                            if is_save:
                                                output_dir = 'result'
                                                os.makedirs(output_dir, exist_ok=True)
                                                # Save original, super-resolution, and frame images
                                                # 원본, 초해상도, 프레임 이미지 저장
                                                output_path = os.path.join(output_dir, f'obj_{resource_manager.n}.jpg')
                                                sr_output_path = os.path.join(output_dir, f'obj_{resource_manager.n}_sr.jpg')
                                                frame_output_path = os.path.join(output_dir, f'frame_{resource_manager.n}.jpg')
                                                cv2.imwrite(output_path, cropped_img)
                                                cv2.imwrite(sr_output_path, sr_img)
                                                cv2.imwrite(frame_output_path, display_frame)
                                                
                                                # Generate image caption and perform search if enabled
                                                # 이미지 캡션 생성 및 검색 수행 (활성화된 경우)
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
                                # Overlay super-resolution image if available
                                # 초해상도 이미지 오버레이 (가능한 경우)        
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
                # 상태 정보 표시
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
                
                # Calculate and display current FPS
                # 현재 FPS 계산 및 표시
                current_fps = 1.0 / (time.time() - frame_start_time + 1e-6)
                cv2.putText(display_frame, f"Processing Speed: {int(current_fps)}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error in main loop: {e}")
            cleanup_tracking()
            break
        
        # Display the frame
        # 프레임 표시
        cv2.imshow('Tracking', display_frame)
        
        # Calculate appropriate wait time
        # 적절한 대기 시간 계산
        processing_time = time.time() - frame_start_time
        wait_time = max(1, int((frame_delay - processing_time) * 1000))
        
        # Handle key events
        # 키 이벤트 처리
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f'): # Toggle fixed mode / 고정 모드 토글
            resource_manager.fixed_mode = not resource_manager.fixed_mode
            print(f"Switched to {'fixed' if resource_manager.fixed_mode else 'real-time'} mode")
        elif key == ord('s'): # Toggle search mode / 검색 모드 토글
            search_mode = resource_manager.toggle_search_mode()
            print(f"Search mode {'enabled' if search_mode else 'disabled'}")
        elif key == ord('t'): # Toggle STT recording / 음성 인식 녹음 토글
            if not stt_overlay.is_recording:
                print("녹음을 시작합니다...")
                stt_overlay.start_recording(duration=5)
            else:
                print("녹음을 중지합니다.")
                stt_overlay.stop_recording()
        elif key == ord('c'): # Clean up tracking / 추적 정리
            cleanup_tracking()
        elif key == 32:  # Space - Toggle pause / 스페이스바 - 일시정지 토글
            is_paused = resource_manager.toggle_pause()
            print(f"Video {'paused' if is_paused else 'resumed'}")

    # Cleanup and release resources
    # 정리 및 리소스 해제
    cleanup_tracking()
    stt_overlay.cleanup()
    cap.release()
    cv2.destroyAllWindows()

def video_path_or_index(arg):
    try:
        # Try converting to int for camera index
        return int(arg)
    except ValueError:
        # If not an integer, treat as file path
        return str(arg)

if __name__ == "__main__":
    # Parse command line arguments
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Object tracking system with video input')
    parser.add_argument('--video', type=video_path_or_index, default=0,
                   help='Camera index (0, 1, ...) or path to video file (default: 0 for webcam)')
    parser.add_argument('--save', action='store_true', default=True,
                      help='Save the processed images (default: True)')
    parser.add_argument('--bing', type=str, default=None,
                      help='enter bing search api key (default: None)')
    
    args = parser.parse_args()
    main(video_path=args.video, is_save=args.save, bing_api=args.bing)