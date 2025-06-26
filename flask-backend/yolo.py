import torch
import numpy as np
import cv2
import os
import logging
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOv5Detector:
    def __init__(self, crops_folder: str = 'crops', conf_threshold: float = 0.25, targetClass: int = 0):
        self.model = self.load_model()
        self.crops_folder = crops_folder
        self.conf_threshold = conf_threshold
        self.targetClass = targetClass

        os.makedirs(self.crops_folder, exist_ok=True)
        self.previous_boxes: List[np.ndarray] = []  # Store previous bounding boxes to avoid duplicates

    def load_model(self):
        logging.info("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)
        model.eval()
        logging.info("Model loaded successfully.")
        return model

    def run_inference(self, image_path):
        return self.model(image_path)

    def center_distance(self, box1, box2):
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2
        return np.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)

    def is_redundant(self, new_box, dist_thresh=50):
        for old_box in self.previous_boxes:
            if self.center_distance(old_box, new_box) < dist_thresh:
                return True
        return False

    def save_crop(self, image, det, image_name, suffix=""):
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = image[y1:y2, x1:x2]
        class_name = self.model.names[int(cls)]
        filename = f"{os.path.splitext(image_name)[0]}_{class_name}_{suffix}_{conf:.2f}.jpg"
        path = os.path.join(self.crops_folder, filename)
        cv2.imwrite(path, crop)
        return crop

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image {image_path}")
            return

        image_name = os.path.basename(image_path)
        results = self.run_inference(image_path)
        detections = results.xyxy[0].cpu().numpy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != self.targetClass or conf < self.conf_threshold:
                continue
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:  # Skip very small detections
                continue
            new_box = [x1, y1, x2, y2]
            if self.is_redundant(new_box):
                continue

            self.save_crop(image, det, image_name, suffix=idx)
            self.previous_boxes.append(new_box)  # Track to avoid duplicates

    def process_folder(self, folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        logging.info(f"Found {len(files)} images in {folder}")
        for file in tqdm(files, desc="Processing images", unit="image"):
            self.process_image(os.path.join(folder, file))
        logging.info(f"Saved all detected crops to {self.crops_folder}")

# Example Usage:
detector = YOLOv5Detector(crops_folder="final_crops", targetClass=0)  # Class 0 = person
detector.process_folder("input_images")