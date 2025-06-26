import os
import shutil
from glob import glob
from yolo import YOLOv5Detector 
from compare import load_and_embed, CosineSimilarity 
import capture

# Class name to YOLO class index mapping
lookup_class = {"humans": 0, "bikes": 3, "cars": 2}

# Directory constants
FRAMES_DIR = "frames"
CANDIDATES_DIR = "candidates"
REFERENCE_DIR = "reference"
REFERENCE_CROPS_DIR = "reference_crops"

def generate_frames(video_path):
    capture.extract_frames(video_path, FRAMES_DIR)

def process_reference_images(target):
    reference_paths = sorted(glob(os.path.join(REFERENCE_DIR, "*.jpg")))
    if not reference_paths:
        print("[WARN] No reference images found.")
        return

    detector = YOLOv5Detector(output_folder=REFERENCE_CROPS_DIR, targetClass=lookup_class[target])
    print(f"[INFO] Processing {len(reference_paths)} reference images...")

    for ref in reference_paths:
        detector.process_image(ref)

    print(f"[INFO] Reference crops saved to {REFERENCE_CROPS_DIR}/")

def process_all_frames_sequentially(target):
    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("[WARN] No frames found to process.")
        return

    detector = YOLOv5Detector(output_folder=CANDIDATES_DIR, targetClass=lookup_class[target])
    print(f"[INFO] Processing {len(frame_paths)} video frames...")

    for frame in frame_paths:
        detector.process_image(frame)

    print(f"[INFO] Frame crops saved to {CANDIDATES_DIR}/")

def find_best_matches(top_k=3):
    reference_images = sorted(glob(os.path.join(REFERENCE_CROPS_DIR, "*.jpg")))
    candidate_images = sorted(glob(os.path.join(CANDIDATES_DIR, "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images found.")
        return

    ref_embeds = [load_and_embed(ref) for ref in reference_images]
    all_scores = []

    for cand in candidate_images:
        print(f"[INFO] Comparing candidate: {cand}")
        cand_embed = load_and_embed(cand)
        scores = [CosineSimilarity(dim=1)(cand_embed, ref).item() for ref in ref_embeds]
        max_score = max(scores)
        all_scores.append((cand, max_score))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[RESULTS] Top {top_k} Matches:")
    for i, (img_path, score) in enumerate(all_scores[:top_k]):
        print(f"#{i+1}: {img_path} with similarity score: {score:.4f}")

def cleanup():
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(CANDIDATES_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_CROPS_DIR, ignore_errors=True)

if __name__ == "__main__":
    target = "cars"  # Choose from "humans", "bikes", "cars"
    video_path = "input_video.mp4"

    generate_frames(video_path)
    process_reference_images(target)
    process_all_frames_sequentially(target)
    find_best_matches(top_k=3)
    cleanup()
