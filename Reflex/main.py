from Reflex.modules.automatic_tracker import run_yolo_tracker
from Reflex.modules.manual_tracker import run_manual_forehead_tracker


YELLOW = "\033[93m"
RESET = "\033[0m"


def _ask_camera():
    raw = input(f"  {YELLOW}Webcam index{RESET} [default=0]: ").strip()
    return int(raw) if raw.isdigit() else 0


def run(mode):
    cam = _ask_camera()

    if mode == "automatic":
        run_yolo_tracker(camera_index=cam, mode="default")
    elif mode == "manual":
        run_manual_forehead_tracker(camera_index=cam)
    else:
        raise ValueError("Mode must be 'automatic' or 'manual'.")