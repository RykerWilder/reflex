# Reflex
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Shell Script](https://img.shields.io/badge/Shell_Script-121011?style=flat&logo=gnu-bash&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=ultralytics&logoColor=black)

<img src="https://github.com/RykerWilder/static_files/blob/main/reflex.webp" alt="Reflex">

Reflex is a Python computer vision tool that uses a webcam to detect people and place a red crosshair on the subject’s forehead in real time.

The application tracks the target as the person moves and displays visual overlays such as bounding boxes, tracking information, and FPS.

## Features

- Real-time webcam-based person detection.
- Forehead targeting with a red crosshair.
- Automatic and manual startup modes.
- On-screen overlays for boxes, labels, and FPS.
- Screenshot capture during execution.

## Requirements

- Python 3.10 or higher.
- A working webcam.
- macOS, Linux, or Windows with OpenCV support.
- Internet connection only the first time if extra dependencies need to be resolved.

## Installation

1. Clone repository
```bash
git clone https://github.com/RykerWilder/reflex
```

2. Change directory
```bash
cd reflex
```

3. Start setup.sh
```bash
bash setup.sh
```

## Usage

Show CLI help:

```bash
reflex --help
```

Start in automatic mode:

```bash
reflex -a
```

Start in manual mode:

```bash
reflex -m
```

When the program starts, you will be asked for the webcam index:

```text
Webcam index [default=0]:
```
Press Enter to use the default webcam.

## Controls for automatic

- `S` saves a screenshot.
- `Q` or `ESC` closes the application.

## Notes

- The YOLO models are loaded from `Reflex/models/`.
- Screenshots are saved during runtime in the screenshots directory used by the application.

## License

This project is distributed under the terms of the license included in the `LICENSE` file.