import os
import shutil
from glob import glob
from yolo import YOLOv5Matcher 
from compare import load_and_embed, CosineSimilarity 
import capture


lookup_class = {"humans": 0, "bikes": 3, "cars": 2}
FRAMES_DIR = "frames"
CANDIDATES_DIR = "candidates"
REFERENCE_DIR = "reference"

def generate_frames(video_path):
    capture.extract_frames(video_path, FRAMES_DIR)

def process_all_frames_sequentially(target):
    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("[WARN] No frames found to process.")
        return
    matcher = YOLOv5Matcher(ref_folder=REFERENCE_DIR, output_folder=CANDIDATES_DIR, targetClass=lookup_class[target])
    print(f"[INFO] Processing {len(frame_paths)} frames sequentially...")

    for frame in frame_paths:
        matcher.process_image(frame)

    print(f"[INFO] YOLO processing complete. Matched frames stored in {CANDIDATES_DIR}/")

def find_best_matches(top_k=3):
    reference_images = sorted(glob(os.path.join(REFERENCE_DIR, "*")))
    candidate_images = sorted(glob(os.path.join("crops", "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images found.")
        return

    ref_embeds = [load_and_embed(ref) for ref in reference_images]
    all_scores = []

    for cand in candidate_images:
        print(f"Processing {cand}")
        cand_embed = load_and_embed(cand)
        scores = [CosineSimilarity(dim=1)(cand_embed, ref).item() for ref in ref_embeds]
        max_score = max(scores)
        all_scores.append((cand, max_score))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n[RESULTS] Top {top_k} Matches:")
    for i, (img_path, score) in enumerate(all_scores[:top_k]):
        print(f"#{i+1}: Frame {img_path} with similarity score: {score:.4f}")

def cleanup():
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)

