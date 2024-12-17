# Scouter Project
AI-powered object detection system with real-time enhancement and accessibility features

## Overview
An AI system inspired by Cyberpunk 2077 that combines several advanced features:
- Zero-shot object detection with detailed captioning
- Real-time object enhancement and upscaling
- Speech-to-text (STT) functionality for accessibility
- Overlay display system similar to cyberpunk HUD elements

## Demo Screenshots
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0c5e8926-57a7-42c0-bf76-7619836c0aa4" width="400" alt="Demo Screenshot 1"/></td>
    <td><img src="https://github.com/user-attachments/assets/b24d2dc9-c3c8-4f19-a334-56ba1f9823e8" width="400" alt="Demo Screenshot 2"/></td>
  </tr>
</table>

## Installation and Setup

1. Clone the repository
```bash
git clone https://github.com/MinwooKim1990/Scouter_project.git
cd Scouter_project
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
python system.py
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

### Quick Reference
| Action | Key/Button | Function |
|--------|------------|----------|
| Select Object | Left Click | Activates detection & tracking |
| Release Tracking | Right Click | Stops tracking current object |
| Toggle Upscaling | F | Enhances tracked object quality |
| Voice Recording | T | Start/Stop subtitle recording |

## Features
- Zero-shot object detection
- Real-time object tracking
- AI-powered image upscaling
- Speech-to-text subtitle generation
- Cyberpunk-style overlay display

## Models & Attributions
This project utilizes several open-source models:

- **MobileSAM**: Zero-shot object detection - Apache-2.0 License
  - Source: [Ultralytics MobileSAM](https://github.com/ultralytics/ultralytics)
- **Fast-SRGAN**: Real-time image upscaling - MIT License
  - Source: Original Fast-SRGAN implementation
- **OpenAI Whisper**: Speech-to-text processing - MIT License
  - Source: [OpenAI Whisper](https://github.com/openai/whisper)
- **Florence 2**: Image captioning - MIT License
  - Source: [Microsoft Florence](https://github.com/microsoft/florence)

## License
This project is licensed under the MIT License since all major components use either MIT or Apache-2.0 licenses. The MIT License is compatible with both and maintains the open-source nature of the utilized models.

See the [LICENSE](LICENSE) file for details.