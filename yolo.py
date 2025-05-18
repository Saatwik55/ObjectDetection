
import torch
import numpy as np
import cv2
import os
import logging
import time
from typing import List
from IPython.display import Image, display
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOv5CarMatcher:
    def __init__(self, ref_folder: str, output_folder: str = 'output', crops_folder: str = 'crops', conf_threshold: float = 0.25):
        self.model = self.load_model()
        self.ref_histograms, self.ref_shapes = self.load_reference_features(ref_folder)
        self.output_folder = output_folder
        self.crops_folder = crops_folder
        self.conf_threshold = conf_threshold
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.crops_folder, exist_ok=True)

    def load_model(self):
        logging.info("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)
        model.eval()
        logging.info("Model loaded successfully.")
        return model

    def load_reference_features(self, ref_folder):
        histograms = []
        shapes = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        for file in os.listdir(ref_folder):
            if file.lower().endswith(supported_formats):
                img_path = os.path.join(ref_folder, file)
                img = cv2.imread(img_path)
                if img is not None:
                    histograms.append(self.get_color_histogram(img))
                    shapes.append(self.get_shape_descriptor(img))
        if not histograms:
            raise ValueError("No valid reference images found.")
        logging.info(f"Loaded {len(histograms)} reference features.")
        return histograms, shapes

    def get_color_histogram(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def get_shape_descriptor(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(thresh)
        hu_moments = cv2.HuMoments(moments).flatten()
        return hu_moments

    def is_similar(self, crop, hist_threshold=0.7, shape_threshold=0.002):
        if crop is None or crop.size == 0:
            return False
        crop_hist = self.get_color_histogram(crop)
        crop_shape = self.get_shape_descriptor(crop)

        for ref_hist, ref_shape in zip(self.ref_histograms, self.ref_shapes):
            hist_score = cv2.compareHist(ref_hist, crop_hist, cv2.HISTCMP_CORREL)
            shape_score = np.sum(np.abs(crop_shape - ref_shape))
            if shape_score < shape_threshold and hist_score < hist_threshold:
                return True
        return False

    def run_inference(self, image_path):
        logging.info(f"Running inference on {image_path}...")
        return self.model(image_path)

    def apply_nms(self, detections, iou_threshold=0.4):
        if len(detections) == 0:
            return detections
        boxes = detections[:, :4]
        scores = detections[:, 4]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=self.conf_threshold, nms_threshold=iou_threshold)
        return detections[indices.flatten()] if len(indices) > 0 else []

    def augment_image(self, image):
        """Unused in test, but kept for potential training use."""
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
        if random.random() < 0.5:
            angle = random.randint(-30, 30)
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return image

    def save_crops(self, image, detections, image_name):
        """Save cropped objects based on detection results."""
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Skip small bounding boxes
            w, h = x2 - x1, y2 - y1
            if w < 30 or h < 30:
                continue
                
            # Crop the object
            cropped_object = image[y1:y2, x1:x2]
            
            # Create a filename
            class_name = self.model.names[int(cls)]
            filename = f"{os.path.splitext(image_name)[0]}_{class_name}_{i}_{conf:.2f}.jpg"
            crop_path = os.path.join(self.crops_folder, filename)
            
            # Save the cropped image
            cv2.imwrite(crop_path, cropped_object)
            logging.info(f"Saved crop: {crop_path}")

    def process_image(self, image_path):
        start_time = time.time()
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Failed to load image {image_path}")
            return
        image_orig = image.copy()
        # image = self.augment_image(image)  # <- Removed from testing

        results = self.run_inference(image_path)
        detections = results.xyxy[0].cpu().numpy()
        
        # Save all detected objects as crops
        image_name = os.path.basename(image_path)
        self.save_crops(image_orig, detections, image_name)
        
        # Apply NMS for display and matching
        detections = self.apply_nms(detections)
        match_found = False

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 2 and conf >= self.conf_threshold:  # Class 2 is car in COCO
                w, h = x2 - x1, y2 - y1
                if w < 30 or h < 30:
                    continue
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                crop = image[y1:y2, x1:x2]
                if self.is_similar(crop):
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Match: {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    match_found = True

        if match_found:
            out_path = os.path.join(self.output_folder, f"matched_{os.path.basename(image_path)}")
            cv2.imwrite(out_path, image)
            logging.info(f"Match found and saved: {out_path}")
            display(Image(filename=out_path))
        else:
            blank_image = np.ones_like(image) * 255
            cv2.putText(blank_image, "No Match Found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            out_path = os.path.join(self.output_folder, f"no_match_{os.path.basename(image_path)}")
            cv2.imwrite(out_path, blank_image)
            logging.info(f"No match found. Saved 'No Match' image: {out_path}")
            display(Image(filename=out_path))

        elapsed = time.time() - start_time
        logging.info(f"Processed {os.path.basename(image_path)} in {elapsed:.2f} seconds")

    def process_folder(self, folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        logging.info(f"Found {len(files)} images in {folder}")
        start_time = time.time()
        total_images = len(files)
        match_count = 0

        for file in tqdm(files, desc="Processing images", unit="image"):
            path = os.path.join(folder, file)
            self.process_image(path)
            match_count += 1

        elapsed = time.time() - start_time
        avg_time = elapsed / total_images
        logging.info(f"Processed all images in {elapsed:.2f} seconds")
        logging.info(f"Average time per image: {avg_time:.2f} seconds")
        logging.info(f"Total matches found: {match_count}")

if __name__ == '__main__':
    input_folder = "frames"
    reference_folder = "reference"
    matcher = YOLOv5CarMatcher(ref_folder=reference_folder, conf_threshold=0.3)
    matcher.process_folder(input_folder)