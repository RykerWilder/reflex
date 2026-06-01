import cv2


try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT_BG = (0, 0, 0)


COCO_CLASSES = {
    "all":        None,
    "person":     0,
    "bicycle":    1,
    "car":        2,
    "motorcycle": 3,
    "bus":        5,
    "truck":      7,
    "cat":        15,
    "dog":        16,
    "bottle":     39,
    "cup":        41,
    "laptop":     63,
    "phone":      67,
}
CLASS_NAMES = list(COCO_CLASSES.keys())


_ID_COLORS = [
    (0, 255, 80),   (255, 180, 0),  (0, 180, 255),  (255, 0, 180),
    (80, 255, 255), (255, 80, 80),  (80, 80, 255),  (180, 255, 0),
]


def _get_color(track_id):
    return _ID_COLORS[int(track_id) % len(_ID_COLORS)]


def _idx_to_name_map():
    return {v: k for k, v in COCO_CLASSES.items() if v is not None}


def _active_classes_to_label(active_classes):
    if not active_classes:
        return "ALL"
    idx_to_name = _idx_to_name_map()
    names = [idx_to_name.get(i, str(i)) for i in sorted(active_classes)]
    return ", ".join(name.upper() for name in names)


def _draw_crosshair(frame, x1, y1, x2, y2, color, track_id, label, conf):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    arm = min(20, w // 4, h // 4)
    corner = min(16, w // 5, h // 5)
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


def _overlay_text(frame, text, pos, color, scale=0.65, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, COLOR_TEXT_BG, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _draw_hud(frame, active_classes, fps, n_targets):
    h_frame, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    grey = (200, 200, 200)
    cls_label = _active_classes_to_label(active_classes)

    _overlay_text(frame, "MODE    : YOLOv8 Auto Tracker", (8, 22), grey, 0.5, 1)
    _overlay_text(frame, f"CLASS   : {cls_label}", (8, 44), (0, 220, 255))
    _overlay_text(frame, f"TARGETS : {n_targets}", (8, 66), (0, 255, 80) if n_targets else grey)
    _overlay_text(frame, f"FPS     : {fps:.1f}", (8, 88), grey, 0.5, 1)

    guide = "[C] Change Targets  |  [Q] Exit"
    _overlay_text(frame, guide, (8, h_frame - 10), grey, 0.42, 1)


def _pick_class_menu(active_classes):
    """
    Menu multi-selezione:
    - numero => toggle classe
    - a      => ALL (nessun filtro)
    - done   => conferma
    - Enter  => conferma
    """
    current = set(active_classes)

    while True:
        print("\n─────────────────────────────────────────────────────────")
        print("  Multi-class target selection")
        print("  Toggle classes by typing the number.")
        print("  Commands: [a]=ALL  [done]=confirm  [Enter]=confirm")
        print("─────────────────────────────────────────────────────────")

        for i, name in enumerate(CLASS_NAMES):
            coco_idx = COCO_CLASSES[name]
            if name == "all":
                selected = (len(current) == 0)
            else:
                selected = coco_idx in current
            marker = " ✓" if selected else ""
            print(f"  [{i:2d}] {name}{marker}")

        print("─────────────────────────────────────────────────────────")
        raw = input("  Class Number / Command: ").strip().lower()

        if raw == "" or raw == "done":
            return current

        if raw == "a":
            current.clear()
            continue

        if raw.isdigit() and 0 <= int(raw) < len(CLASS_NAMES):
            idx = int(raw)
            name = CLASS_NAMES[idx]
            coco_idx = COCO_CLASSES[name]

            if name == "all":
                current.clear()
            else:
                if coco_idx in current:
                    current.remove(coco_idx)
                else:
                    current.add(coco_idx)


def run_yolo_tracker(camera_index=0, model_path="yolov8n.pt"):
    if not _YOLO_AVAILABLE:
        print("[ERROR] ultralytics not installed")
        print("Run: pip install ultralytics")
        return

    print(f"\n[YOLOv8] Loading model '{model_path}'...")
    model = YOLO(model_path)
    print("[YOLOv8] Model ready")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Impossible to find cam (index {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Default: person + phone
    active_classes = {COCO_CLASSES["person"], COCO_CLASSES["phone"]}

    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps = 0.0

    print(f"\n[YOLOv8] Tracking started – classes: {_active_classes_to_label(active_classes)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame not received by webcam.")
            break

        curr_tick = cv2.getTickCount()
        fps = tick_freq / (curr_tick - prev_tick + 1e-9)
        prev_tick = curr_tick

        kwargs = dict(persist=True, verbose=False, conf=0.40, iou=0.45)

        # Se active_classes è vuoto => nessun filtro => traccia tutte le classi
        if active_classes:
            kwargs["classes"] = sorted(active_classes)

        results = model.track(frame, **kwargs)
        n_targets = 0

        boxes_data = results[0].boxes
        if boxes_data is not None and boxes_data.id is not None:
            ids = boxes_data.id.cpu().numpy().astype(int)
            xyxys = boxes_data.xyxy.cpu().numpy().astype(int)
            confs = boxes_data.conf.cpu().numpy()
            clss = boxes_data.cls.cpu().numpy().astype(int)

            n_targets = len(ids)

            for tid, (x1, y1, x2, y2), conf, cls_id in zip(ids, xyxys, confs, clss):
                color = _get_color(tid)
                try:
                    lbl = results[0].names[cls_id]
                except (KeyError, IndexError):
                    lbl = str(cls_id)

                _draw_crosshair(frame, x1, y1, x2, y2, color, tid, lbl, conf)

        _draw_hud(frame, active_classes, fps, n_targets)
        cv2.imshow("Smart Tracker – YOLOv8", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            cv2.destroyAllWindows()
            active_classes = _pick_class_menu(active_classes)
            print(f"[YOLOv8] Active classes → {_active_classes_to_label(active_classes)}")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[YOLOv8] Session stopped.")