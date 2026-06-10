from Reflex.modules.yolo_tracker import run_yolo_tracker

YELLOW = "\033[93m"
RESET = "\033[0m"


def _ask_camera():
    raw = input(f"  {YELLOW}Webcam index{RESET} [default=0]: ").strip()
    return int(raw) if raw.isdigit() else 0


def run(mode="default"):
    cam = _ask_camera()
    run_yolo_tracker(camera_index=cam, mode=mode)


def main():
    run(mode="default")


if __name__ == "__main__":
    main()