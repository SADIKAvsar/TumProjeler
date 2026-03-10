import os
import time
import cv2
import mss
import numpy as np
from ultralytics import YOLO
from datetime import datetime

class AesaRadar:
    def __init__(self, model_path='yolov10n.pt'):
        # 🎯 Radarı (YOLOv10) Yükle - CUDA (GPU) desteğiyle
        self.model = YOLO(model_path).to('cuda')
        self.dataset_path = "AESA_Intelligence/Training_Data"
        os.makedirs(self.dataset_path, exist_ok=True)
        print("📡 AESA Radar: Aktif ve RTX 4060 Ti üzerinden tarama yapıyor.")

    def scan_and_record(self, active_boss_name="Unknown"):
        """Ekranı tarar, nesne tespiti yapar ve eğitim verisi kaydeder."""
        with mss.mss() as sct:
            # 2K Monitör Taraması (Gerektiğinde 'monitor' ayarı yapılabilir)
            screenshot = sct.grab(sct.monitors[1])
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 🚀 YOLOv10 ile Anlık Tespit (Inference)
            results = self.model.predict(frame, conf=0.4, verbose=False)

            # 💾 İstihbarat Kaydı (Dataset Oluşturma)
            # Her 2 saniyede bir veya Boss tespit edildiğinde kaydet
            self._save_intelligence(frame, results, active_boss_name)

            return results

    def _save_intelligence(self, frame, results, label):
        """Tespit edilen kareleri YOLO formatında etiketleyerek kaydeder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"{self.dataset_path}/{label}_{timestamp}.jpg"
        txt_filename = f"{self.dataset_path}/{label}_{timestamp}.txt"

        # Görüntüyü kaydet
        cv2.imwrite(img_filename, frame)

        # Etiketleri (Labels) YOLO formatında yaz (class x_center y_center width height)
        with open(txt_filename, "w") as f:
            for result in results:
                for box in result.boxes:
                    # Koordinatları normalize et
                    xywh = box.xywhn[0].tolist() 
                    cls = int(box.cls[0])
                    f.write(f"{cls} {' '.join(map(str, xywh))}\n")

# --- Test Çalıştırması ---
if __name__ == "__main__":
    radar = AesaRadar()
    while True:
        # Mevcut botun kestiği boss ismini buraya dinamik gönderebiliriz
        radar.scan_and_record("Boss_920_Zeus")
        time.sleep(0.5) # Çevreci tarama hızı