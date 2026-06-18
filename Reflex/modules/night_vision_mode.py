import cv2
import numpy as np

class NightVisionMode:
    
    def __init__(self):
        self.enabled = False
        self.night_vision_frame = None
        
    def toggle(self):
        self.enabled = not self.enabled
        status = "attivata" if self.enabled else "disattivata"
        print(f"[NIGHT VISION] Modalità visione notturna {status}")
        return self.enabled
    
    def apply_effect(self, frame):
        if not self.enabled:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        enhanced = cv2.equalizeHist(gray)

        night_vision = np.zeros_like(frame)
        night_vision[:, :, 1] = enhanced  # Canale verde
        
        # Aggiungi un leggero effetto di rumore (opzionale)
        noise = np.random.randint(0, 20, night_vision.shape[:2], dtype=np.uint8)
        night_vision[:, :, 1] = cv2.add(night_vision[:, :, 1], noise)

        h, w = night_vision.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w/3)
        kernel_y = cv2.getGaussianKernel(h, h/3)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = mask[:, :, np.newaxis]
        
        night_vision = (night_vision * mask).astype(np.uint8)

        cv2.putText(night_vision, "NIGHT VISION", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(night_vision, "[N] Disattiva", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.night_vision_frame = night_vision
        return night_vision
    
    def is_enabled(self):
        return self.enabled
    
    def get_status_text(self):
        return "NIGHT VISION: ON" if self.enabled else "NIGHT VISION: OFF"