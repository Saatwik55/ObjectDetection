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
    augment_reference_crops()

def augment_reference_crops():
    print("[INFO] Augmenting reference crops...")
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    crop_paths = [f for f in os.listdir(REFERENCE_CROPS_DIR) if f.lower().endswith(supported_formats)]

    for file in crop_paths:
        if '_gray' in file or '_bright' in file:
            continue  # Skip already-augmented files

        full_path = os.path.join(REFERENCE_CROPS_DIR, file)
        crop = cv2.imread(full_path)
        if crop is None or crop.size == 0:
            continue

        base_name = os.path.splitext(file)[0]

        # Desaturated / Gray
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(REFERENCE_CROPS_DIR, f"{base_name}_gray.jpg"), gray_bgr)

        # Brightened
        bright = cv2.convertScaleAbs(crop, alpha=1.3, beta=30)
        cv2.imwrite(os.path.join(REFERENCE_CROPS_DIR, f"{base_name}_bright.jpg"), bright)

    print("[INFO] Reference crop augmentation complete.")

def process_all_frames():
    frame_paths = sorted(glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("[WARN] No frames found to process.")
        return

    print(f"[INFO] Processing {len(frame_paths)} video frames...")
    for frame in tqdm(frame_paths, desc="YOLO on frames"):
        detector.process_image(frame)

    print(f"[INFO] Frame crops saved to {CANDIDATES_DIR}")

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

