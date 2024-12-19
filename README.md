# Scouter Project v 1.0
AI-powered object detection system with real-time enhancement and accessibility features

#### Contributor:
- **Lead System Developer**: Minwoo Kim - [Minwoo Kim Github](https://github.com/MinwooKim1990)
- **Azure AI Developer**: Duhyeon Nam - [Duhyeon Nam Github](https://github.com/namduhus)
- **AI Code Engineer**: Nakyung Cho - [Nakyung Cho Github](http://github.com/nakyung1007)
- **Azure System Developer**: Seungwoo Hong - [Seungwoo Hong Github](https://github.com/Seungwoo-H1)
- **AI Code Engineer**: Hyunjun Kim - [Hyunjun Kim Github](https://github.com/hyunjun-kim4984)
- **Azure AI Developer**: Sumin Hyun - [Sumin Hyun Github](https://github.com/hellooobella)

## Overview
An AI system inspired by Cyberpunk 2077 that combines several advanced features:
- Zero-shot object detection with detailed captioning
- Real-time object enhancement and upscaling
- Speech-to-text (STT) functionality for accessibility
- Overlay display system similar to Cyberpunk 2077 Scanner elements
- Extra information with Bing Search API (Default: Deactivated)
- STT to LLM Response on display

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
    <td><img src="https://github.com/user-attachments/assets/735d84b8-a426-4614-a867-d36b1c37e6bc" width="400" alt="Demo Screenshot 3"/></td>
    <td><img src="https://github.com/user-attachments/assets/e5db2cd5-2960-46e8-98bb-fc20fe054402" width="400" alt="Demo Screenshot 4"/></td>
  </tr>
</table>

## STT to LLM Demo Screenshots
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/eed1b282-17d3-402b-be88-dfd9e63b588d" width="400" alt="Demo Screenshot 5"/></td>
    <td><img src="https://github.com/user-attachments/assets/715e3da5-1395-45f6-96e7-04b76cef8a7f" width="400" alt="Demo Screenshot 6"/></td>
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
python system.py --video path/to/your/video.mp4(integer 0 for webcam, Default: 0) --bing [YOUR-API-KEY] --llm-provider [google, openai, groq] --llm-api-key [YOUR-API-KEY] --llm-model [providing models]
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

#### STT to LLM
- **A Key**: 
  - First Press: Starts prompt recording for LLM
  - Second Press: Stops prompt recording  and processes the LLM output
<table>
  <tr>
    <th>Provider</th>
    <th>Model</th>
  </tr>
  <tr>
    <td rowspan="4" align="center">Google</td>
    <td>gemini-2.0-flash-exp</td>
  </tr>
  <tr>
    <td>gemini-1.5-flash</td>
  </tr>
  <tr>
    <td>gemini-1.5-flash-8b</td>
  </tr>
  <tr>
    <td>gemini-1.5-pro</td>
  </tr>
  <tr>
    <td rowspan="4" align="center">OpenAI</td>
    <td>gpt-4o-2024-08-06</td>
  </tr>
  <tr>
    <td>gpt-4o-mini-2024-07-18</td>
  </tr>
  <tr>
    <td>o1-2024-12-17</td>
  </tr>
  <tr>
    <td>gpt-3.5-turbo-0125</td>
  </tr>
  <tr>
    <td rowspan="5" align="center">GROQ</td>
    <td>llama-3.3-70b-versatile</td>
  </tr>
  <tr>
    <td>llama-3.2-90b-text-preview</td>
  </tr>
  <tr>
    <td>llama-3.2-11b-text-preview</td>
  </tr>
  <tr>
    <td>gemma2-9b-it</td>
  </tr>
  <tr>
    <td>mixtral-8x7b-32768</td>
  </tr>
</table>

#### Stop Video
- **Space Key**: 
  - First Press: Stop video playing and can do object detection
  - Second Press: play video again

#### Stop Video
- **Tab Key**: 
  - First Press: Turn on full instructions (default: turn off)
  - Second Press: Turn off all instructions

### Quick Reference
| Action | Key/Button | Function |
|--------|------------|----------|
| Select Object | Left Click | Activates detection & tracking |
| Release Tracking | Right Click | Stops tracking current object |
| Toggle Upscaling | F | Enable realtime upscaling |
| Voice Recording | T | Start/Stop subtitle recording |
| Image Search | S | Activate Bing image search |
| STT to LLM | A | Start/Stop prompt recording to LLM |
| Play/Stop | Space | Play/Stop video |
| Show Instructions | Tab | Showing/Not Showing instructions |

## Features
- Zero-shot object detection
- Real-time object tracking
- AI-powered image upscaling
- Speech-to-text subtitle generation
- Speech-to-LLM output generation
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
- **Google Gemini API**: LLM Response
  - Source: [Google AI](https://ai.google.dev/gemini-api/docs/models/gemini)
- **OpenAI API**: LLM Response
  - Source: [OpenAI](https://platform.openai.com/docs/overview)
- **GROQ API**: LLM Response
  - Source: [GROQ](https://console.groq.com/docs/overview)

## License
This project is licensed under the MIT License since all major components use either MIT or Apache-2.0 licenses. The MIT License is compatible with both and maintains the open-source nature of the utilized models.

This project uses the following APIs. Please ensure compliance with their respective terms of use:

- **Bing Search API**:
  - **Documentation**: [Bing Web Search API](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/)
  - **Pricing**: [Bing Search API Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/search-api/)
  - **Licensing**: Usage of the Bing Search API is subject to Microsoft's [Terms of Use](https://www.microsoft.com/en-us/servicesagreement) and [Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement).

- **OpenAI API**:
  - **Documentation**: [OpenAI API](https://platform.openai.com/docs/api-reference)
  - **Terms of Use**: [OpenAI Terms of Use](https://platform.openai.com/terms)
  - **Licensing**: OpenAI's API usage is governed by their [Terms of Use](https://platform.openai.com/terms) and [Usage Policies](https://platform.openai.com/policies/usage-policies).

- **Google GenAI API**:
  - **Documentation**: [Google GenAI API](https://developers.generativeai.google/)
  - **Terms of Service**: [Google API Terms](https://developers.google.com/terms/)
  - **Licensing**: Google's APIs are subject to the [Google APIs Terms of Service](https://developers.google.com/terms/).

- **Groq API**:
  - **Documentation**: [Groq API Documentation](https://console.groq.com/docs)
  - **Licensing**: Usage of the Groq API must comply with Groq's API policies. Specific licensing details can be found in their [API documentation](https://console.groq.com/docs).

### Note
To ensure the security of API keys, store them securely using environment variables or secret management solutions. Do not expose sensitive information in public repositories.

See the [LICENSE](LICENSE) file for details.