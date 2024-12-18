# Scouter Project
AI-powered object detection system with real-time enhancement and accessibility features

## Overview
An AI system inspired by Cyberpunk 2077 that combines several advanced features:
- Zero-shot object detection with detailed captioning
- Real-time object enhancement and upscaling
- Speech-to-text (STT) functionality for accessibility
- Overlay display system similar to Cyberpunk 2077 Scanner elements
- Extra information with Bing Search API (Default: Deactivated)

## Demo Screenshots
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0779a6e0-684d-4cf6-89b1-b21d8f02f4b5" width="400" alt="Demo Screenshot 1"/></td>
    <td><img src="https://github.com/user-attachments/assets/857226e5-17fc-446c-8d08-b207f578cff0" width="400" alt="Demo Screenshot 2"/></td>
  </tr>
</table>

## GUI Demo Screenshots
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/735d84b8-a426-4614-a867-d36b1c37e6bc" width="400" alt="Demo Screenshot 1"/></td>
    <td><img src="https://github.com/user-attachments/assets/e5db2cd5-2960-46e8-98bb-fc20fe054402" width="400" alt="Demo Screenshot 2"/></td>
  </tr>
</table>

## Installation and Setup

1. Clone the repository
```bash
git clone https://github.com/MinwooKim1990/Scouter_PJ.git
cd Scouter_PJ
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare your data
```bash
mkdir data
```
Place your video files in the `data` folder

4. Run the application
```bash
python system.py --video path/to/your/video.mp4(integer 0 for webcam, Default: 0) --bing [YOUR-API-KEY]
```

5. For checking arguments help
```bash
python system.py --help
```

## Use GUI

1. Clone the repository
```bash
git clone https://github.com/MinwooKim1990/Scouter_PJ.git
cd Scouter_PJ
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare your data
```bash
mkdir data
```
Place your video files in the `data` folder

4. Run the application
```bash
python tkinterapp.py
```

## How to Use

### Basic Controls

#### Object Detection & Tracking
- **Left Click**: 
  - Click anywhere on the video to activate zero-shot object detection
  - Click on an object to create a bounding box and start tracking
- **Right Click**: 
  - Releases the current bounding box and stops tracking

#### Image Enhancement
- **F Key**: 
  - Toggles real-time upscaling of the tracked object
  - Default: Disabled

#### Voice Subtitles
- **T Key**: 
  - First Press: Starts voice recording for subtitle generation
  - Second Press: Stops recording and processes the subtitle

#### Image Search
- **S Key**: 
  - Toggles Activate image search of detected object
  - Default: Disabled

#### Stop Video
- **Space Key**: 
  - First Press: Stop video playing and can do object detection (GUI system not work)
  - Second Press: play video again

### Quick Reference
| Action | Key/Button | Function |
|--------|------------|----------|
| Select Object | Left Click | Activates detection & tracking |
| Release Tracking | Right Click | Stops tracking current object |
| Toggle Upscaling | F | Enable realtime upscaling |
| Voice Recording | T | Start/Stop subtitle recording |
| Image Search | S | Activate Bing image search |
| play/stop | Space | play/stop video (GUI system not work) |

## Features
- Zero-shot object detection
- Real-time object tracking
- AI-powered image upscaling
- Speech-to-text subtitle generation
- Cyberpunk-style overlay display

## Technical Specifications
### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support
  - Minimum VRAM: 8GB (For realtime Upscaling: 24GB)
  - Tested on: NVIDIA GPU RTX 4090
- **Memory**: 2GB
- **Storage**: 2.5GB

### Performance Notes
- Tested 720P quality videos
- Processing FPS Without Realtime Upscaling FPS: 30 ~ 40 FPS
- Processing FPS during heaviest work load: 10 ~ 20 FPS
- Current version is optimized for VRAM efficiency
- Peak VRAM usage: ~7GB during operation (In Realtime Upscaling ~ 21GB during operation)
- Further optimization may be available in future updates

## Models & Attributions
This project utilizes several open-source models:

- **MobileSAM**: Zero-shot object detection - Apache-2.0 License
  - Source: [Ultralytics MobileSAM](https://docs.ultralytics.com/ko/models/mobile-sam/)
- **Fast-SRGAN**: Real-time image upscaling - MIT License
  - Source: [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN)
- **OpenAI Whisper**: Speech-to-text processing - MIT License
  - Source: [OpenAI Whisper](https://github.com/openai/whisper)
- **Florence 2**: Image captioning - MIT License
  - Source: [Microsoft Florence](https://huggingface.co/microsoft/Florence-2-large)
- **Bing Search API**: Image Search
  - Source: [Microsoft Azure](https://learn.microsoft.com/en-us/bing/search-apis/bing-image-search/overview)

## License
This project is licensed under the MIT License since all major components use either MIT or Apache-2.0 licenses. The MIT License is compatible with both and maintains the open-source nature of the utilized models.

See the [LICENSE](LICENSE) file for details.