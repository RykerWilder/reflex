import cv2
import math
import time
import subprocess
from datetime import datetime
from pathlib import Path
from .night_vision_mode import NightVisionMode

try:
    from ultralytics import YOLO, settings
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT_BG = (0, 0, 0)
COLOR_FOREHEAD_DOT = (0, 0, 255)

KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4

PERSON_CLASS_ID = 0

_ID_COLORS = [
    (0, 255, 80), (255, 180, 0), (0, 180, 255), (255, 0, 180),
    (80, 255, 255), (255, 80, 80), (80, 80, 255), (180, 255, 0),
]


def _models_dir():
    return Path.cwd() / "Reflex/models"


def _default_model_paths():
    models_dir = _models_dir()
    return str(models_dir / "yolov8n.pt"), str(models_dir / "yolov8n-pose.pt")


def _default_screenshot_dir():
    return str(Path.cwd() / "screenshots")


def _ensure_model_dir(model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)


def _get_color(track_id):
    return _ID_COLORS[int(track_id) % len(_ID_COLORS)]


def _overlay_text(frame, text, pos, color, scale=0.65, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, COLOR_TEXT_BG, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_crosshair(frame, x1, y1, x2, y2, color, track_id, label, conf):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    arm = min(20, max(8, w // 4), max(8, h // 4))
    corner = min(16, max(6, w // 5), max(6, h // 5))
    thick = 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, thick)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, thick)
    cv2.circle(frame, (cx, cy), 4, color, thick)

    for px, py, dx, dy in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * corner, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, thick)

    tag = f"ID:{int(track_id)}  {label}  {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(tag, FONT, 0.52, 1)
    ty = y1 - 6 if y1 - 6 > th else y1 + th + 4
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, tag, (x1 + 2, ty), FONT, 0.52, color, 1, cv2.LINE_AA)


def _draw_object_box(frame, x1, y1, x2, y2, label, conf, color=(180, 180, 180)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"{label} {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(tag, FONT, 0.52, 1)
    ty = y1 - 6 if y1 - 6 > th else y1 + th + 4
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, tag, (x1 + 2, ty), FONT, 0.52, color, 1, cv2.LINE_AA)


def _draw_hud(frame, fps, n_targets, n_objects, mode, night_vision):
    h_frame, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 145), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    grey = (200, 200, 200)
    _overlay_text(frame, f"MODE    : {mode.upper()}", (8, 22), grey, 0.5, 1)
    _overlay_text(frame, f"OBJECTS : {n_objects}", (8, 44), (255, 180, 0))
    _overlay_text(frame, f"PERSONS : {n_targets}", (8, 66), (0, 255, 80) if n_targets else grey)
    _overlay_text(frame, f"FPS     : {fps:.1f}", (8, 88), grey, 0.5, 1)

    nv_color = (0, 255, 0) if night_vision.is_enabled() else grey
    _overlay_text(frame, night_vision.get_status_text(), (8, 110), nv_color, 0.5, 1)
    
    _overlay_text(frame, "[N] Night Vision  [S] Screenshot   [Q] Quit", (8, h_frame - 10), grey, 0.42, 1)


def _valid_point(pt, conf=None, min_conf=0.35):
    x, y = float(pt[0]), float(pt[1])
    if conf is not None and conf < min_conf:
        return False
    return x > 0 and y > 0


def _smooth_point(track_memory, track_id, x, y, alpha=0.35):
    if track_id not in track_memory:
        track_memory[track_id] = (float(x), float(y))
    else:
        px, py = track_memory[track_id]
        sx = alpha * float(x) + (1.0 - alpha) * px
        sy = alpha * float(y) + (1.0 - alpha) * py
        track_memory[track_id] = (sx, sy)
    return int(track_memory[track_id][0]), int(track_memory[track_id][1])


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
    red = COLOR_FOREHEAD_DOT
    cv2.circle(frame, (fx, fy), 4, red, -1)
    cv2.circle(frame, (fx, fy), 11, red, 1)
    cv2.line(frame, (fx - 16, fy), (fx - 8, fy), red, 1)
    cv2.line(frame, (fx + 8, fy), (fx + 16, fy), red, 1)
    cv2.line(frame, (fx, fy - 16), (fx, fy - 8), red, 1)
    cv2.line(frame, (fx, fy + 8), (fx, fy + 16), red, 1)


def _save_screenshot(frame, prefix="screenshot", save_dir=None):
    save_dir = save_dir or _default_screenshot_dir()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    full_path = Path(save_dir) / filename

    ok = cv2.imwrite(str(full_path), frame)
    if ok:
        print(f"[REFLEX] Sreenshot saved: {full_path}")
    else:
        print(f"[REFLEX] Error while saving screenshot to: {full_path}")


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


def _launch_command_for_human(track_id, conf_score, cooldowns, threshold=0.80, cooldown_sec=3.0):
    if conf_score < threshold:
        return

    now = time.time()
    last_time = cooldowns.get(track_id, 0.0)

    if now - last_time < cooldown_sec:
        return

    cooldowns[track_id] = now

    try:
        subprocess.run(
            ["echo", f"Human detected: ID={track_id}, conf={conf_score:.2f}"],
            check=False
        )
        print("Command launched")
    except Exception as e:
        print(f"[COMMAND ERROR] {e}")


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


def run_yolo_tracker(
    camera_index=0,
    mode="default",
    detect_model_path=None,
    pose_model_path=None,
    screenshot_dir=None,
):
    if not _YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed")
        return

    detect_model_path = detect_model_path or _default_model_paths()[0]
    pose_model_path = pose_model_path or _default_model_paths()[1]
    screenshot_dir = screenshot_dir or _default_screenshot_dir()

    print(f"[YOLO] ultralytics weights_dir setting: {settings['weights_dir']}")
    print(f"[YOLO] local detect model path: {detect_model_path}")
    print(f"[YOLO] local pose model path  : {pose_model_path}")

    try:
        det_model = _load_yolo_model(detect_model_path, "detection model")
        pose_model = _load_yolo_model(pose_model_path, "pose model")
    except Exception:
        print("[ERROR] Model loading failed. Check internet connection, model path, or Ultralytics installation.")
        return

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

    forehead_memory = {}
    next_person_id = 0
    command_cooldowns = {}

    print(f"\n[REFLEX] Tracking started in automatic mode.")
    print("[REFLEX] Press 'N' to toggle Night Vision.")
    print("[REFLEX] Press 'S' to save a screenshot.")
    print("[REFLEX] Press 'Q' or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame not received from webcam.")
                break

            # Apply Night Vision if enabled
            if night_vision.is_enabled():
                frame = night_vision.apply_effect(frame)

            curr_tick = cv2.getTickCount()
            fps = tick_freq / (curr_tick - prev_tick + 1e-9)
            prev_tick = curr_tick

            conf = 0.35 if mode == "automatic" else 0.45
            iou = 0.45 if mode == "automatic" else 0.35

            det_results = det_model.track(
                frame,
                persist=True,
                verbose=False,
                conf=conf,
                iou=iou,
            )

            n_objects = 0
            n_targets = 0
            active_ids = set()

            det_boxes = det_results[0].boxes
            if det_boxes is not None and det_boxes.xyxy is not None:
                d_xyxys = det_boxes.xyxy.cpu().numpy().astype(int)
                d_confs = det_boxes.conf.cpu().numpy()
                d_clss = det_boxes.cls.cpu().numpy().astype(int)
                d_ids = det_boxes.id.cpu().numpy().astype(int) if det_boxes.id is not None else None

                n_objects = len(d_xyxys)

                for idx, (box, conf_score, cls_id) in enumerate(zip(d_xyxys, d_confs, d_clss)):
                    x1, y1, x2, y2 = map(int, box)

                    try:
                        lbl = det_results[0].names[cls_id]
                    except (KeyError, IndexError):
                        lbl = str(cls_id)

                    if cls_id == PERSON_CLASS_ID:
                        tid = int(d_ids[idx]) if d_ids is not None else next_person_id + idx
                        active_ids.add(tid)
                        color = _get_color(tid)

                        _launch_command_for_human(
                            tid,
                            float(conf_score),
                            command_cooldowns,
                            threshold=0.80,
                            cooldown_sec=3.0
                        )

                        _draw_crosshair(frame, x1, y1, x2, y2, color, tid, lbl, conf_score)

                        pose_box, person_kpts_xy, person_kpts_conf = _run_pose_on_person_crop(
                            pose_model,
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

                            fx = max(0, min(frame.shape[1] - 1, fx))
                            fy = max(0, min(frame.shape[0] - 1, fy))

                            smooth_alpha = 0.20 if mode == "manual" else 0.35
                            fx, fy = _smooth_point(forehead_memory, tid, fx, fy, alpha=smooth_alpha)
                            _draw_forehead_reticle(frame, fx, fy)

                        n_targets += 1
                    else:
                        if mode == "automatic":
                            _draw_object_box(frame, x1, y1, x2, y2, lbl, conf_score)

                if d_ids is None:
                    next_person_id += len(d_xyxys)

            stale_ids = [tid for tid in list(forehead_memory.keys()) if tid not in active_ids]
            for tid in stale_ids:
                del forehead_memory[tid]

            stale_command_ids = [tid for tid in list(command_cooldowns.keys()) if tid not in active_ids]
            for tid in stale_command_ids:
                del command_cooldowns[tid]

            _draw_hud(frame, fps, n_targets, n_objects, mode, night_vision)
            cv2.imshow("Reflex", frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("n"):
                night_vision.toggle()
                
            elif key == ord("s"):
                _save_screenshot(frame, prefix=f"reflex_{mode}", save_dir=screenshot_dir)

            elif key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[REFLEX] Session stopped.")