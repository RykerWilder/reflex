import cv2
import math
import time
from pathlib import Path
from datetime import datetime
from .night_vision_mode import NightVisionMode

try:
    from ultralytics import YOLO, settings
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT_BG = (0, 0, 0)
COLOR_BOX = (0, 255, 255)
COLOR_TEXT = (220, 220, 220)
COLOR_FOREHEAD = (0, 0, 255)
COLOR_TRACKING = (0, 255, 120)
COLOR_SELECTION = (0, 200, 255)

KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4

PERSON_CLASS_ID = 0


def _models_dir():
    return Path.cwd() / "Reflex/models"


def _default_model_paths():
    models_dir = _models_dir()
    return str(models_dir / "yolov8n.pt"), str(models_dir / "yolov8n-pose.pt")


def _default_screenshot_dir():
    return str(Path.cwd() / "screenshots")


def _ensure_model_dir(model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)


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
        print(f"[REFLEX] Screenshot saved: {path}")


def _create_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise AttributeError("CSRT tracker not found")


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


def _estimate_forehead_from_pose(kpts_xy, kpts_conf, box):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    nose = kpts_xy[KP_NOSE]
    left_eye = kpts_xy[KP_LEFT_EYE]
    right_eye = kpts_xy[KP_RIGHT_EYE]
    left_ear = kpts_xy[KP_LEFT_EAR]
    right_ear = kpts_xy[KP_RIGHT_EAR]

    nose_c = kpts_conf[KP_NOSE] if kpts_conf is not None else None
    left_eye_c = kpts_conf[KP_LEFT_EYE] if kpts_conf is not None else None
    right_eye_c = kpts_conf[KP_RIGHT_EYE] if kpts_conf is not None else None
    left_ear_c = kpts_conf[KP_LEFT_EAR] if kpts_conf is not None else None
    right_ear_c = kpts_conf[KP_RIGHT_EAR] if kpts_conf is not None else None

    def _valid_point(pt, conf=None, min_conf=0.35):
        x, y = float(pt[0]), float(pt[1])
        if conf is not None and conf < min_conf:
            return False
        return x > 0 and y > 0

    left_eye_ok = _valid_point(left_eye, left_eye_c)
    right_eye_ok = _valid_point(right_eye, right_eye_c)
    nose_ok = _valid_point(nose, nose_c)
    left_ear_ok = _valid_point(left_ear, left_ear_c)
    right_ear_ok = _valid_point(right_ear, right_ear_c)

    if left_eye_ok and right_eye_ok:
        ex = (left_eye[0] + right_eye[0]) / 2.0
        ey = (left_eye[1] + right_eye[1]) / 2.0
        eye_dist = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
        return int(ex), int(ey - max(10.0, eye_dist * 0.55))

    if nose_ok and left_eye_ok:
        dx = left_eye[0] - nose[0]
        dy = left_eye[1] - nose[1]
        return int(nose[0] + dx * 0.5), int(nose[1] - max(10.0, abs(dy) * 2.2, h * 0.10))

    if nose_ok and right_eye_ok:
        dx = right_eye[0] - nose[0]
        dy = right_eye[1] - nose[1]
        return int(nose[0] + dx * 0.5), int(nose[1] - max(10.0, abs(dy) * 2.2, h * 0.10))

    if left_ear_ok and right_ear_ok:
        return int((left_ear[0] + right_ear[0]) / 2.0), int((left_ear[1] + right_ear[1]) / 2.0 - h * 0.10)

    return int((x1 + x2) / 2), int(y1 + h * 0.16)


def _draw_forehead_reticle(frame, fx, fy):
    cv2.circle(frame, (fx, fy), 4, COLOR_FOREHEAD, -1)
    cv2.circle(frame, (fx, fy), 11, COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx - 16, fy), (fx - 8, fy), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx + 8, fy), (fx + 16, fy), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx, fy - 16), (fx, fy - 8), COLOR_FOREHEAD, 1)
    cv2.line(frame, (fx, fy + 8), (fx, fy + 16), COLOR_FOREHEAD, 1)


def _draw_target_box(frame, box, show_forehead=True, fx=None, fy=None):
    x, y, w, h = [int(v) for v in box]
    x2, y2 = x + w, y + h
    cx, cy = x + w // 2, y + h // 2

    cv2.rectangle(frame, (x, y), (x2, y2), COLOR_BOX, 2)
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), COLOR_BOX, 1)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), COLOR_BOX, 1)
    cv2.circle(frame, (cx, cy), 4, COLOR_BOX, 1)

    if show_forehead and fx is not None and fy is not None:
        cv2.circle(frame, (fx, fy), 3, COLOR_FOREHEAD, -1)
        cv2.line(frame, (fx - 8, fy), (fx + 8, fy), COLOR_FOREHEAD, 1)
        cv2.line(frame, (fx, fy - 8), (fx, fy + 8), COLOR_FOREHEAD, 1)

    label = f"TARGET  {w}x{h}"
    (tw, th), bl = cv2.getTextSize(label, FONT, 0.52, 1)
    ty = y - 6 if y - 6 > th else y + th + 4
    cv2.rectangle(frame, (x, ty - th - 2), (x + tw + 4, ty + bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, label, (x + 2, ty), FONT, 0.52, COLOR_BOX, 1, cv2.LINE_AA)


def _run_pose_on_person_crop(pose_model, frame, person_box, pose_conf=0.35):
    x1, y1, x2, y2 = map(int, person_box)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None, None

    pose_results = pose_model(crop, verbose=False, conf=pose_conf)
    if not pose_results or pose_results[0].boxes is None or pose_results[0].keypoints is None:
        return None, None, None

    p_boxes = pose_results[0].boxes
    p_kpts = pose_results[0].keypoints

    if p_boxes.xyxy is None or p_kpts.xy is None:
        return None, None, None

    boxes_xyxy = p_boxes.xyxy.cpu().numpy()
    confs = p_boxes.conf.cpu().numpy() if p_boxes.conf is not None else None
    kpts_xy = p_kpts.xy.cpu().numpy()
    kpts_conf = p_kpts.conf.cpu().numpy() if p_kpts.conf is not None else None

    if len(boxes_xyxy) == 0 or len(kpts_xy) == 0:
        return None, None, None

    best_idx = int(confs.argmax()) if confs is not None and len(confs) > 0 else 0

    pose_box = boxes_xyxy[best_idx]
    pose_kpts_xy = kpts_xy[best_idx]
    pose_kpts_conf = kpts_conf[best_idx] if kpts_conf is not None else None

    pose_box_global = (
        int(pose_box[0] + x1),
        int(pose_box[1] + y1),
        int(pose_box[2] + x1),
        int(pose_box[3] + y1),
    )

    pose_kpts_xy_global = pose_kpts_xy.copy()
    pose_kpts_xy_global[:, 0] += x1
    pose_kpts_xy_global[:, 1] += y1

    return pose_box_global, pose_kpts_xy_global, pose_kpts_conf


def _load_yolo_model(model_name_or_path, label):
    try:
        _ensure_model_dir(model_name_or_path)
        print(f"[YOLO] Loading {label}: {model_name_or_path}")
        model = YOLO(model_name_or_path)
        print(f"[YOLO] {label} ready")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load {label}: {model_name_or_path}")
        print(f"[ERROR] {e}")
        raise


class MouseSelector:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.selection_complete = False
        self.roi = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.end_point = None
            self.selecting = True
            self.selection_complete = False
            self.roi = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting and self.start_point is not None:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selecting and self.start_point is not None:
                self.end_point = (x, y)
                self.selecting = False
                self.selection_complete = True

                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])

                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.roi = (x1, y1, x2 - x1, y2 - y1)
                    print(f"[REFLEX] Selection made: {self.roi}")
                else:
                    self.roi = None
                    print("[REFLEX] Selection too small, please select a larger area")

    def get_selection(self):
        roi = self.roi
        self.roi = None
        self.selection_complete = False
        return roi

    def draw_selection(self, frame):
        if self.selecting and self.start_point is not None and self.end_point is not None:
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SELECTION, 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), COLOR_SELECTION, 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), COLOR_SELECTION, 1)

            w = x2 - x1
            h = y2 - y1
            size_text = f"{w}x{h}"
            cv2.putText(frame, size_text, (x1, y1 - 10), FONT, 0.5, COLOR_SELECTION, 1)


class ManualAITracker:
    def __init__(self):
        self.tracker = None
        self.tracking = False
        self.bbox = None
        self.forehead_memory = None
        self.pose_model = None
        self.det_model = None

    def initialize_ai_models(self, detect_model_path=None, pose_model_path=None):
        if not _YOLO_AVAILABLE:
            print("[ERROR] ultralytics not installed")
            return False

        detect_model_path = detect_model_path or _default_model_paths()[0]
        pose_model_path = pose_model_path or _default_model_paths()[1]

        try:
            self.det_model = _load_yolo_model(detect_model_path, "detection model")
            self.pose_model = _load_yolo_model(pose_model_path, "pose model")
            print("[REFLEX] AI models loaded successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load AI models: {e}")
            return False

    def get_forehead_from_ai(self, frame, bbox):
        if self.pose_model is None:
            return None

        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        pose_box, person_kpts_xy, person_kpts_conf = _run_pose_on_person_crop(
            self.pose_model,
            frame,
            (x1, y1, x2, y2),
            pose_conf=0.35
        )

        if person_kpts_xy is not None:
            px1, py1, px2, py2 = pose_box if pose_box is not None else (x1, y1, x2, y2)
            fx, fy = _estimate_forehead_from_pose(
                person_kpts_xy,
                person_kpts_conf,
                (px1, py1, px2, py2)
            )
            return fx, fy

        return None

    def get_forehead_from_bbox(self, bbox):
        return _estimate_forehead_from_bbox(bbox)

    def update(self, frame):
        if not self.tracking or self.tracker is None:
            return False, None

        ok, new_box = self.tracker.update(frame)

        if ok:
            self.bbox = new_box

            fx, fy = None, None
            ai_result = self.get_forehead_from_ai(frame, new_box)
            if ai_result is not None:
                fx, fy = ai_result

            if fx is None or fy is None:
                fx, fy = self.get_forehead_from_bbox(new_box)

            fx = max(0, min(frame.shape[1] - 1, fx))
            fy = max(0, min(frame.shape[0] - 1, fy))

            fx, fy = _smooth_point(self.forehead_memory, fx, fy, alpha=0.28)
            self.forehead_memory = (fx, fy)

            return True, (fx, fy)
        else:
            self.tracking = False
            self.tracker = None
            self.bbox = None
            self.forehead_memory = None
            return False, None

    def start_tracking(self, frame, roi):
        try:
            self.tracker = _create_tracker()
            ok = self.tracker.init(frame, roi)
            if ok:
                self.bbox = roi
                self.tracking = True
                self.forehead_memory = None
                print(f"[REFLEX] Target selected: {roi}")
                return True
            else:
                self.tracker = None
                self.tracking = False
                self.bbox = None
                self.forehead_memory = None
                print("[REFLEX] Init failed.")
                return False
        except Exception as e:
            print(f"[REFLEX] Error starting tracker: {e}")
            return False

    def reset(self):
        self.tracker = None
        self.tracking = False
        self.bbox = None
        self.forehead_memory = None
        print("[REFLEX] Reset target.")


def run_manual_forehead_tracker(
    camera_index=0,
    detect_model_path=None,
    pose_model_path=None,
    screenshot_dir=None
):
    
    screenshot_dir = screenshot_dir or _default_screenshot_dir()

    tracker = ManualAITracker()
    if not tracker.initialize_ai_models(detect_model_path, pose_model_path):
        print("[ERROR] Failed to initialize AI models. Exiting.")
        return

    mouse_selector = MouseSelector()

    night_vision = NightVisionMode()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Impossible to open cam (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0

    window_name = "Reflex AI Forehead Tracker"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_selector.mouse_callback)

    print("[REFLEX] Click and drag on the person to select target.")
    print("[REFLEX] Press N to toggle Night Vision.")
    print("[REFLEX] Press R to reset tracker.")
    print("[REFLEX] Press S to save screenshot.")
    print("[REFLEX] Press Q or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame not received from webcam.")
                break

            if night_vision.is_enabled():
                frame = night_vision.apply_effect(frame)

            curr_tick = cv2.getTickCount()
            fps = tick_freq / (curr_tick - prev_tick + 1e-9)
            prev_tick = curr_tick

            roi = mouse_selector.get_selection()
            if roi is not None:
                tracker.start_tracking(frame, roi)

            mouse_selector.draw_selection(frame)

            ok, forehead_pos = tracker.update(frame)

            if ok and forehead_pos is not None:
                fx, fy = forehead_pos
                _draw_target_box(frame, tracker.bbox, show_forehead=True, fx=fx, fy=fy)
                _draw_forehead_reticle(frame, fx, fy)
                _overlay_text(frame, "STATUS: TRACKING", (10, 25), COLOR_TRACKING, 0.55, 1)
                _overlay_text(frame, f"FOREHEAD: ({fx}, {fy})", (10, 50), COLOR_FOREHEAD, 0.55, 1)
            else:
                if tracker.tracking:
                    _overlay_text(frame, "STATUS: LOST", (10, 25), (0, 0, 255), 0.55, 1)
                else:
                    _overlay_text(frame, "STATUS: IDLE - Click & Drag to select", (10, 25), (180, 180, 180), 0.55, 1)

            _overlay_text(frame, f"FPS: {fps:.1f}", (10, 75), COLOR_TEXT, 0.55, 1)
            _overlay_text(frame, night_vision.get_status_text(), (10, 125), (0, 255, 0) if night_vision.is_enabled() else (180, 180, 180), 0.45, 1)
            _overlay_text(frame, "  [N] Night Vision  [R] Reset  [S] Screenshot  [Q] Quit", 
                         (10, frame.shape[0] - 12), COLOR_TEXT, 0.45, 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                night_vision.toggle()

            elif key == ord("r"):
                tracker.reset()

            elif key == ord("s"):
                _save_screenshot(frame, save_dir=screenshot_dir)

            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[REFLEX] Session stopped.")


if __name__ == "__main__":
    run_manual_forehead_tracker()