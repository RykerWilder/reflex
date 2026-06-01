"""
YOLOv8 Tracker Module
Rilevamento e tracciamento automatico tramite Ultralytics YOLOv8 + BoT-SORT.
Il modello yolov8n.pt (~6 MB) viene scaricato automaticamente al primo avvio.

Tasto C  -> cambia classe COCO da tracciare
Tasto Q  -> esci
"""

import cv2

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

FONT          = cv2.FONT_HERSHEY_SIMPLEX
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
    (80, 255, 255), (255, 80, 80),  (80, 80, 255),   (180, 255, 0),
]


def _get_color(track_id):
    return _ID_COLORS[int(track_id) % len(_ID_COLORS)]


def _draw_crosshair(frame, x1, y1, x2, y2, color, track_id, label, conf):
    cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
    w, h    = x2 - x1, y2 - y1
    arm     = min(20, w // 4, h // 4)
    corner  = min(16, w // 5, h // 5)
    thick   = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), color, thick)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), color, thick)
    cv2.circle(frame, (cx, cy), 4, color, thick)
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx * corner, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, thick)
    tag = f"ID:{int(track_id)}  {label}  {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(tag, FONT, 0.52, 1)
    ty = y1 - 6 if y1 - 6 > th else y1 + th + 4
    cv2.rectangle(frame, (x1, ty - th - 2), (x1 + tw + 4, ty + bl), COLOR_TEXT_BG, -1)
    cv2.putText(frame, tag, (x1 + 2, ty), FONT, 0.52, color, 1, cv2.LINE_AA)


def _overlay_text(frame, text, pos, color, scale=0.65, thickness=2):
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), FONT, scale, COLOR_TEXT_BG, thickness+1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),     FONT, scale, color,          thickness,   cv2.LINE_AA)


def _draw_hud(frame, class_name, fps, n_targets, class_idx):
    h_frame, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 100), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    grey = (200, 200, 200)
    _overlay_text(frame, "MODE    : YOLOv8 Auto Tracker", (8, 22), grey, 0.5, 1)
    _overlay_text(frame, f"CLASSE  : {class_name.upper()} (idx {class_idx if class_idx is not None else '*'})",
                  (8, 44), (0, 220, 255))
    _overlay_text(frame, f"TARGETS : {n_targets}", (8, 66), (0, 255, 80) if n_targets else grey)
    _overlay_text(frame, f"FPS     : {fps:.1f}",   (8, 88), grey, 0.5, 1)
    guide = "[C] Cambia classe  |  [Q] Esci"
    _overlay_text(frame, guide, (8, h_frame - 10), grey, 0.42, 1)


def _pick_class_menu(current_idx):
    print("\n─── Scegli la classe da tracciare ───────────────────────")
    for i, name in enumerate(CLASS_NAMES):
        marker = " ◄" if i == current_idx else ""
        print(f"  [{i:2d}] {name}{marker}")
    print("─────────────────────────────────────────────────────────")
    raw = input("  Numero classe (Invio = annulla): ").strip()
    if raw.isdigit() and 0 <= int(raw) < len(CLASS_NAMES):
        idx  = int(raw)
        name = CLASS_NAMES[idx]
        return name, COCO_CLASSES[name]
    return CLASS_NAMES[current_idx], COCO_CLASSES[CLASS_NAMES[current_idx]]


def run_yolo_tracker(camera_index=0, model_path="yolov8n.pt"):
    """Avvia il tracker YOLOv8 sulla webcam specificata."""
    if not _YOLO_AVAILABLE:
        print("[ERRORE] La libreria 'ultralytics' non è installata.")
        print("         Esegui:  pip install ultralytics")
        return

    print(f"\n[YOLOv8] Caricamento modello '{model_path}'...")
    model = YOLO(model_path)
    print("[YOLOv8] Modello caricato.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire la webcam (indice {camera_index}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    class_name_idx = CLASS_NAMES.index("person")
    class_name     = CLASS_NAMES[class_name_idx]
    class_filter   = COCO_CLASSES[class_name]

    tick_freq = cv2.getTickFrequency()
    prev_tick = cv2.getTickCount()
    fps       = 0.0

    print(f"\n[YOLOv8] Tracciamento avviato – classe: {class_name.upper()}")
    print("          Premi C nella finestra per cambiare classe | Q per uscire.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVVISO] Frame non ricevuto dalla webcam.")
            break

        curr_tick = cv2.getTickCount()
        fps       = tick_freq / (curr_tick - prev_tick + 1e-9)
        prev_tick = curr_tick

        kwargs = dict(persist=True, verbose=False, conf=0.40, iou=0.45)
        if class_filter is not None:
            kwargs["classes"] = [class_filter]

        results   = model.track(frame, **kwargs)
        n_targets = 0

        boxes_data = results[0].boxes
        if boxes_data is not None and boxes_data.id is not None:
            ids   = boxes_data.id.cpu().numpy().astype(int)
            xyxys = boxes_data.xyxy.cpu().numpy().astype(int)
            confs = boxes_data.conf.cpu().numpy()
            clss  = boxes_data.cls.cpu().numpy().astype(int)
            n_targets = len(ids)
            for tid, (x1, y1, x2, y2), conf, cls_id in zip(ids, xyxys, confs, clss):
                color = _get_color(tid)
                try:
                    lbl = model.names[cls_id]
                except (KeyError, IndexError):
                    lbl = str(cls_id)
                _draw_crosshair(frame, x1, y1, x2, y2, color, tid, lbl, conf)

        _draw_hud(frame, class_name, fps, n_targets, class_filter)
        cv2.imshow("Smart Tracker – YOLOv8", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            cv2.destroyAllWindows()
            class_name, class_filter = _pick_class_menu(class_name_idx)
            class_name_idx = CLASS_NAMES.index(class_name)
            print(f"[YOLOv8] Classe cambiata → {class_name.upper()}")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[YOLOv8 Tracker] Sessione terminata.")