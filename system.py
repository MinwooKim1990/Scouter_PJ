import os
import cv2
import time
import torch
import argparse
import numpy as np
from ultralytics import SAM
from utils.overlayed_display import ImageOverlay
from utils.whisper_overlay import WhisperSTTOverlay
from utils.srprocessor import SuperResolutionProcessor
from utils.caption import generate_caption_async, CaptionState

# ResourceManager: Class to manage system resources and tracking states
# 시스템 자원과 추적 상태를 관리하는 클래스
class ResourceManager:
    def __init__(self):
        self.click_point = None  # Store mouse click coordinates / 마우스 클릭 좌표 저장
        self.tracking_started = False  # Flag for tracking status / 추적 시작 여부 플래그
        self.initial_detection = True  # Flag for first detection / 첫 감지 여부 플래그
        self.fixed_mode = True  # Toggle between fixed and real-time mode / 고정 모드와 실시간 모드 전환
        self.n = 1  # Counter for saved images / 저장된 이미지 카운터
        self.current_results = None  # Store current detection results / 현재 감지 결과 저장
        
    # Clean up GPU memory and reset results
    # GPU 메모리 정리 및 결과 초기화
    def cleanup_resources(self):
        if self.current_results is not None:
            del self.current_results
            self.current_results = None
        torch.cuda.empty_cache()

def main(video_path ='data/squirrel.mp4', is_save = True):
    # Initialize core components
    # 핵심 컴포넌트 초기화  
    resource_manager = ResourceManager()
    IO = ImageOverlay()  # Handle image overlay operations / 이미지 오버레이 처리
    min_size = 256
    min_size = 256
    max_size = 512
    sr_processor = SuperResolutionProcessor(min_size, max_size)  # Super resolution processor / 초해상도 처리기
    
    torch.cuda.empty_cache()  # Clear GPU memory / GPU 메모리 정리
    
    # Load SAM (Segment Anything Model) for object detection
    # 객체 감지를 위한 SAM 모델 로드
    with torch.cuda.device(0):
        model = SAM('models/mobile_sam.pt')

    # Set up video capture
    # 비디오 캡처 설정
    video_path = video_path
    cap = cv2.VideoCapture(video_path)

    # Initialize speech-to-text overlay
    # 음성-텍스트 변환 오버레이 초기화
    stt_overlay = WhisperSTTOverlay(model_type="tiny")
    
    # Get video properties
    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = 1 / fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up display window and mouse callback
    # 디스플레이 창 및 마우스 콜백 설정
    cv2.namedWindow('Tracking')
    cv2.setMouseCallback('Tracking', IO.mouse_callback)
    
    fixed_sr_img = None
    last_cleanup_time = time.time()
    
    def cleanup_tracking():
        # Function to clean up resources and reset tracking state
        # 자원을 정리하고 추적 상태를 초기화하는 함수
        nonlocal fixed_sr_img
        resource_manager.cleanup_resources()

        # Delete super resolution image if it exists
        # 초해상도 이미지가 존재하면 삭제
        if fixed_sr_img is not None:
            del fixed_sr_img
            fixed_sr_img = None

        # Reset all tracking states to initial values
        # 모든 추적 상태를 초기값으로 재설정
        IO.set_tracking_state(
            sr_img=None,
            should_show_overlay=False,
            click_point=None,
            tracking_started=False,
            initial_detection=True
        )
        # Clear GPU memory / GPU 메모리 정리
        torch.cuda.empty_cache()

    # Main processing loop
    # 메인 처리 루프
    while cap.isOpened():
        try:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            tracking_state = IO.get_tracking_state()

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
                # When tracking is active and click point exists
                # 추적이 활성화되고 클릭 포인트가 있을 때
                if tracking_state['click_point'] is not None:
                    if resource_manager.current_results is not None:
                        del resource_manager.current_results
                    # Perform object detection using SAM
                    # SAM을 사용한 객체 감지 수행
                    with torch.no_grad():
                        resource_manager.current_results = model.predict(
                            frame, 
                            points=[tracking_state['click_point']], 
                            labels=[1]
                        )
                    
                    results = resource_manager.current_results
                    # If object detected, process the results
                    # 객체가 감지되면 결과 처리
                    if len(results) > 0 and len(results[0].masks) > 0:
                        mask = results[0].masks.data[0].cpu().numpy()
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                        # Process largest contour
                        # 가장 큰 윤곽선 처리
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            bbox = cv2.boundingRect(largest_contour)
                            x, y, w, h = bbox
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Check if super resolution processing is needed
                            # 초해상도 처리가 필요한지 확인
                            should_process_sr = tracking_state['initial_detection'] or (not resource_manager.fixed_mode and tracking_state['prev_bbox'] != bbox)
                            
                            if should_process_sr:
                                # Crop the detected object with padding
                                # 감지된 객체를 패딩과 함께 잘라내기
                                cropped_img = IO.crop_object(frame, bbox, padding=20)
                                try:
                                    # Clean up previous super resolution image in real-time mode
                                    # 실시간 모드에서 이전 초해상도 이미지 정리
                                    if fixed_sr_img is not None and not resource_manager.fixed_mode:
                                        del fixed_sr_img
                                    
                                    # Process image with super resolution
                                    # 초해상도로 이미지 처리
                                    sr_img = sr_processor.process_image(cropped_img)
                                    IO.set_tracking_state(
                                        sr_img=sr_img,
                                        should_show_overlay=True,
                                        prev_bbox=bbox
                                    )

                                    # Handle initial detection case
                                    # 최초 감지 시 처리
                                    if tracking_state['initial_detection']:
                                        # Create and position info panel
                                        # 정보 패널 생성 및 위치 지정
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

                                        # Save images if save mode is enabled
                                        # 저장 모드가 활성화된 경우 이미지 저장
                                        if is_save:
                                            output_dir = 'result'
                                            os.makedirs(output_dir, exist_ok=True)
                                            output_path = os.path.join(output_dir, f'obj_{resource_manager.n}.jpg')
                                            sr_output_path = os.path.join(output_dir, f'obj_{resource_manager.n}_sr.jpg')
                                            frame_output_path = os.path.join(output_dir, f'frame_{resource_manager.n}.jpg')
                                            cv2.imwrite(output_path, cropped_img)
                                            cv2.imwrite(sr_output_path, sr_img)
                                            cv2.imwrite(frame_output_path, display_frame)
                                            
                                            # Generate caption for the saved image
                                            # 저장된 이미지에 대한 캡션 생성
                                            caption_state = CaptionState()
                                            caption_state.start_processing()
                                            generate_caption_async(sr_output_path)
                                            
                                            fixed_sr_img = sr_img.copy()
                                            print(f"Saved: {output_path} and {sr_output_path}")
                                            resource_manager.n += 1
                                    elif not resource_manager.fixed_mode:
                                        # Update super resolution image in real-time mode
                                        # 실시간 모드에서 초해상도 이미지 업데이트
                                        fixed_sr_img = sr_img.copy()
                                    
                                except Exception as e:
                                    print(f"Error processing image: {e}")
                                    IO.set_tracking_state(
                                        sr_img=None,
                                        should_show_overlay=False
                                    )
                            # Check if overlay should be displayed
                            # 오버레이를 표시해야 하는지 확인        
                            if tracking_state['should_show_overlay'] and tracking_state['panel_position'] is not None:
                                display_frame = IO.overlay_image(
                                    display_frame,
                                    fixed_sr_img,
                                    tracking_state['panel_position'][0],
                                    tracking_state['panel_position'][1],
                                    original_bbox_size=(w, h),
                                    bbox=(x, y, w, h)
                                )

                            # Update click point for continuous tracking
                            # 연속 추적을 위한 클릭 포인트 업데이트    
                            IO.click_point = (x + w//2, y + h//2)
                
                cv2.putText(display_frame, "Right click to cancel tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Press 'f' to {'disable' if resource_manager.fixed_mode else 'enable'} fixed mode", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Mode: {'Fixed' if resource_manager.fixed_mode else 'Real-time'}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Calculate and display processing speed
                # 처리 속도 계산 및 표시
                current_fps = 1.0 / (time.time() - frame_start_time + 1e-6)
                cv2.putText(display_frame, f"Processing Speed: {int(current_fps)}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in main loop: {e}")
            cleanup_tracking()
            break

        # Show the frame and handle keyboard input
        # 프레임 표시 및 키보드 입력 처리
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
        elif key == ord('c'):
            cleanup_tracking()

    # Cleanup and release resources
    # 정리 및 자원 해제
    cleanup_tracking()
    stt_overlay.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object tracking system with video input')
    parser.add_argument('--video', type=str, default='data/squirrel.mp4',
                      help='Path to the video file (default: data/squirrel.mp4)')
    parser.add_argument('--save', action='store_true', default=True,
                      help='Save the processed images (default: True)')
    
    args = parser.parse_args()
    main(video_path=args.video, is_save=args.save)