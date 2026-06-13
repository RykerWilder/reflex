import cv2
import time
from pathlib import Path
from datetime import datetime

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT_BG = (0, 0, 0)
COLOR_BOX = (0, 255, 255)
COLOR_TEXT = (220, 220, 220)
COLOR_FOREHEAD = (0, 0, 255)

tracker = None
tracking = False
bbox = None
forehead_memory = None


def _default_screenshot_dir():
    return str(Path.cwd() / "screenshots")


def _overlay_text(frame, text, pos, color, scale=0.6, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, COLOR_TEXT_BG, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _save_screenshot(frame, prefix="manual_forehead_tracker", save_dir=None):
    save_dir = save_dir or _default_screenshot_dir()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(save_dir) / f"{prefix}_{timestamp}.png"
    ok = cv2.imwrite(str(path), frame)
    if ok:
        print(f"[SCREENSHOT] Saved: {path}")


def _create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise AttributeError(
        "CSRT tracker non disponibile. Installa opencv-contrib-python."
    )


def _smooth_point(prev_point, x, y, alpha=0.30):
    if prev_point is None:
        return int(x), int(y)
    px, py = prev_point
    sx = alpha * float(x) + (1.0 - alpha) * px
    sy = alpha * float(y) + (1.0 - alpha) * py
    return int(sx), int(sy)


def _estimate_forehead_from_bbox(box):
    x, y, w, h = [float(v) for v in box]

    fx = x + w * 0.50
    fy = y + h * 0.18

    return int(fx), int(fy)


def _draw_forehead_reticle(frame, fx, fy):
    cv2.circle(frame, (fx, fy), 4, COLOR_FOREHEAD, -1)
    cv2.circle(frame, (fx, fy), 11, COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx - 16, fy), (fx - 8, fy), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx + 8, fy), (fx + 16, fy), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx, fy - 16), (fx, fy - 8), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx, fy + 8), (fx, fy + 16), COLOR_FOREHEAD, 1)


def _draw_target_box(frame, box):
    x, y, w, h = [int(v) for v in box]
    x2, y2 = x + w, y + h
    cx, cy = x + w // 2, y + h // 2

    cv2.rectangle(frame, (x, y), (x2, y2), COLOR_BOX, 2)
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), COLOR_BOX, 1)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), COLOR_BOX, 1)
    cv2.circle(frame, (cx, cy), 4, COLOR_BOX, 1)

    label = f"TARGET  {w}x{h}"
    (tw, th), bl = cv2.getTextSize(label, FONT, 0.52, 1)
    ty = y - 6 if y - 6 > th else y + th + 4
    cv2.rectangle(frame, (x, ty - th - 2), (x + tw + 4, ty + bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, label, (x + 2, ty), FONT, 0.52, COLOR_BOX, 1, cv2.LINE_AA)


def run_manual_forehead_tracker(camera_index=0, screenshot_dir=None):
    global tracker, tracking, bbox, forehead_memory

    screenshot_dir = screenshot_dir or _default_screenshot_dir()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Impossible to open cam (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0

    window_name = "Reflex Manual Forehead Tracker"

    print("[INFO] Press T to select target with ROI.")
    print("[INFO] Press R to reset tracker.")
    print("[INFO] Press S to save screenshot.")
    print("[INFO] Press Q or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame not received from webcam.")
                break

            curr_tick = cv2.getTickCount()
            fps = tick_freq / (curr_tick - prev_tick + 1e-9)
            prev_tick = curr_tick

            if tracking and tracker is not None:
                ok, new_box = tracker.update(frame)

                if ok:
                    bbox = new_box
                    _draw_target_box(frame, bbox)

                    fx, fy = _estimate_forehead_from_bbox(bbox)
                    fx = max(0, min(frame.shape[1] - 1, fx))
                    fy = max(0, min(frame.shape[0] - 1, fy))

                    fx, fy = _smooth_point(forehead_memory, fx, fy, alpha=0.28)
                    forehead_memory = (fx, fy)

                    _draw_forehead_reticle(frame, fx, fy)

                    _overlay_text(frame, "STATUS: TRACKING", (10, 25), (0, 255, 120), 0.55, 1)
                    _overlay_text(frame, f"FOREHEAD: ({fx}, {fy})", (10, 50), COLOR_FOREHEAD, 0.55, 1)
                else:
                    tracking = False
                    tracker = None
                    bbox = None
                    forehead_memory = None
                    _overlay_text(frame, "STATUS: LOST", (10, 25), (0, 0, 255), 0.55, 1)
            else:
                _overlay_text(frame, "STATUS: IDLE", (10, 25), (180, 180, 180), 0.55, 1)

            _overlay_text(frame, f"FPS: {fps:.1f}", (10, 75), COLOR_TEXT, 0.55, 1)
            _overlay_text(frame, "[T] Select ROI  [R] Reset  [S] Screenshot  [Q] Quit", (10, frame.shape[0] - 12), COLOR_TEXT, 0.45, 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("t"):
                frozen = frame.copy()
                roi = cv2.selectROI(window_name, frozen, fromCenter=False, showCrosshair=True)
                if roi != (0, 0, 0, 0):
                    tracker = _create_tracker()
                    ok = tracker.init(frame, roi)
                    if ok:
                        bbox = roi
                        tracking = True
                        forehead_memory = None
                        print(f"[TRACKER] Target selected: {roi}")
                    else:
                        tracker = None
                        tracking = False
                        bbox = None
                        forehead_memory = None
                        print("[TRACKER] Init failed.")

            elif key == ord("r"):
                tracker = None
                tracking = False
                bbox = None
                forehead_memory = None
                print("[TRACKER] Reset.")

            elif key == ord("s"):
                _save_screenshot(frame, save_dir=screenshot_dir)

            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[TRACKER] Session stopped.")


if __name__ == "__main__":
    run_manual_forehead_tracker()