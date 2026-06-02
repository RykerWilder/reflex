import cv2
import math
from datetime import datetime
import os

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False



FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT_BG = (0, 0, 0)
COLOR_FOREHEAD_DOT = (0, 0, 255)


# COCO pose face keypoints
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4



_ID_COLORS = [
    (0, 255, 80),   (255, 180, 0),  (0, 180, 255),  (255, 0, 180),
    (80, 255, 255), (255, 80, 80),  (80, 80, 255),  (180, 255, 0),
]



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



def _draw_hud(frame, fps, n_targets):
    h_frame, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


    grey = (200, 200, 200)
    _overlay_text(frame, "MODE    : YOLOv8 Pose Tracker", (8, 22), grey, 0.5, 1)
    _overlay_text(frame, "TARGET  : PERSON / FOREHEAD", (8, 44), (0, 220, 255))
    _overlay_text(frame, f"TARGETS : {n_targets}", (8, 66), (0, 255, 80) if n_targets else grey)
    _overlay_text(frame, f"FPS     : {fps:.1f}", (8, 88), grey, 0.5, 1)
    _overlay_text(frame, "[S] Screenshot   [Q] Quit", (8, h_frame - 10), grey, 0.42, 1)



def _valid_point(pt, conf=None, min_conf=0.35):
    x, y = float(pt[0]), float(pt[1])
    if conf is not None and conf < min_conf:
        return False
    return x > 0 and y > 0



def _smooth_point(track_memory, track_id, x, y, alpha=0.35):
    """
    Exponential moving average for a steadier forehead reticle.
    """
    if track_id not in track_memory:
        track_memory[track_id] = (float(x), float(y))
    else:
        px, py = track_memory[track_id]
        sx = alpha * float(x) + (1.0 - alpha) * px
        sy = alpha * float(y) + (1.0 - alpha) * py
        track_memory[track_id] = (sx, sy)
    return int(track_memory[track_id][0]), int(track_memory[track_id][1])



def _estimate_forehead_from_pose(kpts_xy, kpts_conf, box):
    """
    Forehead estimation priority:
    1) both eyes -> midpoint + upward offset
    2) nose + one eye -> upward offset from upper face direction
    3) ears midpoint -> slight upward offset
    4) fallback on box upper-center
    """
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
        forehead_x = ex
        forehead_y = ey - max(10.0, eye_dist * 0.55)
        return int(forehead_x), int(forehead_y)


    if nose_ok and left_eye_ok:
        dx = left_eye[0] - nose[0]
        dy = left_eye[1] - nose[1]
        forehead_x = nose[0] + dx * 0.5
        forehead_y = nose[1] - max(10.0, abs(dy) * 2.2, h * 0.10)
        return int(forehead_x), int(forehead_y)


    if nose_ok and right_eye_ok:
        dx = right_eye[0] - nose[0]
        dy = right_eye[1] - nose[1]
        forehead_x = nose[0] + dx * 0.5
        forehead_y = nose[1] - max(10.0, abs(dy) * 2.2, h * 0.10)
        return int(forehead_x), int(forehead_y)


    if left_ear_ok and right_ear_ok:
        hx = (left_ear[0] + right_ear[0]) / 2.0
        hy = (left_ear[1] + right_ear[1]) / 2.0 - h * 0.10
        return int(hx), int(hy)


    return int((x1 + x2) / 2), int(y1 + h * 0.16)



def _draw_forehead_reticle(frame, fx, fy):
    red = COLOR_FOREHEAD_DOT


    cv2.circle(frame, (fx, fy), 4, red, -1)
    cv2.circle(frame, (fx, fy), 11, red, 1)


    cv2.line(frame, (fx - 16, fy), (fx - 8, fy), red, 1)
    cv2.line(frame, (fx + 8, fy), (fx + 16, fy), red, 1)
    cv2.line(frame, (fx, fy - 16), (fx, fy - 8), red, 1)
    cv2.line(frame, (fx, fy + 8), (fx, fy + 16), red, 1)



def _save_screenshot(frame, prefix="screenshot", save_dir="screenshots"):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    full_path = os.path.join(save_dir, filename)

    ok = cv2.imwrite(full_path, frame)
    if ok:
        print(f"[SCREENSHOT] Saved: {full_path}")
    else:
        print(f"[SCREENSHOT] Error while saving screenshot to: {full_path}")



def run_yolo_tracker(camera_index=0, model_path="yolov8n-pose.pt"):
    if not _YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed")
        print("Run: pip install ultralytics")
        return


    print(f"\n[YOLOv8-Pose] Loading model '{model_path}'...")
    model = YOLO(model_path)
    print("[YOLOv8-Pose] Model ready")


    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Impossible to find cam (index {camera_index}).")
        return


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0


    # track_id -> smoothed forehead point
    forehead_memory = {}


    print("\n[YOLOv8-Pose] Tracking started – forehead reticle enabled")
    print("[YOLOv8-Pose] Press 'S' to save a screenshot.")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame not received by webcam.")
            break


        curr_tick = cv2.getTickCount()
        fps = tick_freq / (curr_tick - prev_tick + 1e-9)
        prev_tick = curr_tick


        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.45,
            iou=0.45,
            classes=[0],   # person only
        )


        n_targets = 0
        active_ids = set()


        boxes_data = results[0].boxes
        keypoints_data = results[0].keypoints


        if (
            boxes_data is not None
            and boxes_data.id is not None
            and keypoints_data is not None
            and keypoints_data.xy is not None
        ):
            ids = boxes_data.id.cpu().numpy().astype(int)
            xyxys = boxes_data.xyxy.cpu().numpy().astype(int)
            confs = boxes_data.conf.cpu().numpy()
            clss = boxes_data.cls.cpu().numpy().astype(int)
            kpts_xy = keypoints_data.xy.cpu().numpy()


            kpts_conf = None
            if keypoints_data.conf is not None:
                kpts_conf = keypoints_data.conf.cpu().numpy()


            n_targets = len(ids)


            for i, (tid, box, conf, cls_id) in enumerate(zip(ids, xyxys, confs, clss)):
                active_ids.add(int(tid))
                x1, y1, x2, y2 = map(int, box)
                color = _get_color(tid)


                try:
                    lbl = results[0].names[cls_id]
                except (KeyError, IndexError):
                    lbl = str(cls_id)


                _draw_crosshair(frame, x1, y1, x2, y2, color, tid, lbl, conf)


                person_kpts_xy = kpts_xy[i]
                person_kpts_conf = kpts_conf[i] if kpts_conf is not None else None


                fx, fy = _estimate_forehead_from_pose(
                    person_kpts_xy,
                    person_kpts_conf,
                    (x1, y1, x2, y2)
                )


                fx = max(0, min(frame.shape[1] - 1, fx))
                fy = max(0, min(frame.shape[0] - 1, fy))


                fx, fy = _smooth_point(forehead_memory, int(tid), fx, fy, alpha=0.35)
                _draw_forehead_reticle(frame, fx, fy)


        
        stale_ids = [tid for tid in forehead_memory.keys() if tid not in active_ids]
        for tid in stale_ids:
            del forehead_memory[tid]


        _draw_hud(frame, fps, n_targets)

        cv2.imshow("Smart Tracker – YOLOv8 Pose Forehead", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            _save_screenshot(frame, prefix="smart_tracker")

        if key in (ord("q"), 27):
            break


    cap.release()
    cv2.destroyAllWindows()
    print("[YOLOv8-Pose] Session stopped.")