import os
import shutil
from glob import glob
import cv2
from tqdm import tqdm
from yolo import YOLOv5Detector
from compare import load_and_embed, CosineSimilarity
import capture

# Configurable constants
lookup_class = {"humans": 0, "bikes": 3, "cars": 2}
FRAMES_DIR = "frames"
CANDIDATES_DIR = "candidates"
REFERENCE_DIR = "reference"
REFERENCE_CROPS_DIR = "reference_crops"
OUTPUT_DIR = "output"
detector = None
# Initialize detector globally
def initialize(target_class_name):
    global detector
    target_class_index = lookup_class[target_class_name]
    detector = YOLOv5Detector(output_folder=CANDIDATES_DIR, targetClass=target_class_index, conf_threshold=0.10)

def generate_frames(video_path):
    capture.extract_frames(video_path, FRAMES_DIR)
    
def process_reference_images():
    print(f"[INFO] Processing reference images in {REFERENCE_DIR}")
    detector.process_reference_folder(ref_folder=REFERENCE_DIR, reference_crops_folder=REFERENCE_CROPS_DIR)
    print(f"[INFO] Reference crops saved to {REFERENCE_CROPS_DIR}")

def process_all_frames():
    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("[WARN] No frames found to process.")
        return

    print(f"[INFO] Processing {len(frame_paths)} video frames...")
    for frame in tqdm(frame_paths, desc="YOLO on frames"):
        detector.process_image(frame)

    print(f"[INFO] Frame crops saved to {CANDIDATES_DIR}")

import shutil

def find_best_matches(top_k=10):
    reference_images = sorted(glob(os.path.join(REFERENCE_CROPS_DIR, "*.jpg")))
    candidate_images = sorted(glob(os.path.join(CANDIDATES_DIR, "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load reference embeddings
    ref_embeds = [load_and_embed(ref) for ref in reference_images]
    all_scores = []

    # Score each candidate
    for cand in candidate_images:
        print(f"[INFO] Comparing candidate: {cand}")
        cand_embed = load_and_embed(cand)
        scores = [CosineSimilarity(dim=1)(cand_embed, ref).item() for ref in ref_embeds]
        max_score = max(scores)
        all_scores.append((cand, max_score))

    # Sort and pick top_k
    all_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = all_scores[:top_k]

    # Copy top matches to output folder with rank-based filename
    for rank, (img_path, _) in enumerate(top_matches, start=1):
        original_filename = os.path.basename(img_path)
        new_filename = f"{rank:02d}_{original_filename}"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, new_filename))

    print(f"[INFO] Top {len(top_matches)} matches saved to '{OUTPUT_DIR}/'")


def cleanup():
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(CANDIDATES_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_CROPS_DIR, ignore_errors=True)

