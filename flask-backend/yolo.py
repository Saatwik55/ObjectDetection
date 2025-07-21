import os
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import re

class YOLODetector:
    def __init__(self, output_folder, conf_threshold=0.40, targetClass=0, buffer_interval=30, buffer_unit='frame'):
        self.model = YOLO("yolov8s.pt")
        self.output_folder = output_folder
        self.conf_threshold = conf_threshold
        self.targetClass = targetClass
        self.buffer_interval = buffer_interval
        self.buffer_unit = buffer_unit
        self.last_saved_info = {}
        os.makedirs(self.output_folder, exist_ok=True)

    def save_crop(self, image, box, cls, conf, obj_id, image_name):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        class_name = self.model.names[int(cls)]
        filename = f"{os.path.splitext(image_name)[0]}_{class_name}_id{obj_id}_{conf:.2f}.jpg"
        path = os.path.join(self.output_folder, filename)
        cv2.imwrite(path, crop)

    def _parse_value(self, image_name):
        if self.buffer_unit == 'frame':
            match = re.search(r'frame_(\d+)', image_name)
            return int(match.group(1)) if match else None
        else:
            match = re.search(r'_sec(\d+)', image_name)
            return int(match.group(1)) if match else None

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return

        image_name = os.path.basename(image_path)
        current_value = self._parse_value(image_name)
        results = self.model.track(source=image_path, tracker="botsort.yaml", persist=True, verbose=False)[0]

        if results is None or results.boxes is None:
            return

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls != self.targetClass or conf < self.conf_threshold:
                continue

            obj_id = int(box.id[0]) if box.id is not None else -1
            if obj_id == -1:
                continue

            xyxy = box.xyxy[0].tolist()
            
            if current_value is None:
                if obj_id not in self.last_saved_info:
                    self.save_crop(image, xyxy, cls, conf, obj_id, image_name)
                    self.last_saved_info[obj_id] = -1
            else:
                if obj_id not in self.last_saved_info:
                    self.save_crop(image, xyxy, cls, conf, obj_id, image_name)
                    self.last_saved_info[obj_id] = current_value
                else:
                    last_val = self.last_saved_info[obj_id]
                    if last_val == -1:
                        continue
                    if current_value >= last_val + self.buffer_interval:
                        self.save_crop(image, xyxy, cls, conf, obj_id, image_name)
                        self.last_saved_info[obj_id] = current_value

    def process_folder(self, folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for file in tqdm(files, desc="Processing images", unit="image"):
            self.process_image(os.path.join(folder, file))

    def process_reference_folder(self, ref_folder, reference_crops_folder):
        os.makedirs(reference_crops_folder, exist_ok=True)
        files = [f for f in os.listdir(ref_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        for file in tqdm(files, desc="Cropping reference images", unit="image"):
            img_path = os.path.join(ref_folder, file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            results = self.model.track(source=img_path, tracker="botsort.yaml", persist=True, verbose=False)[0]
            if results is None or results.boxes is None:
                continue

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != self.targetClass or conf < self.conf_threshold:
                    continue

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                crop = image[y1:y2, x1:x2]
                class_name = self.model.names[int(cls)]
                filename = f"{os.path.splitext(file)[0]}_{class_name}_ref_{conf:.2f}.jpg"
                crop_path = os.path.join(reference_crops_folder, filename)
                cv2.imwrite(crop_path, crop)
                break