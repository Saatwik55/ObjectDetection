import os
import shutil
from glob import glob
from tqdm import tqdm
from torch.nn import CosineSimilarity
from PIL import Image
from yolo import YOLODetector
from compare import load_and_embed
from face import embed_face
import capture

# Configurable constants
lookup_class = {"humans": 0, "bikes": 3, "cars": 2}
FRAMES_DIR = "frames"
CANDIDATES_DIR = "candidates"
REFERENCE_DIR = "reference"
REFERENCE_CROPS_DIR = "reference_crops"
OUTPUT_DIR = "output"
detector = None

def is_image_too_small(image_path, model_type):
    try:
        img = Image.open(image_path)
        width, height = img.size
        if model_type == "dino":
            return width < 128 or height < 128
        elif model_type == "face":
            return width < 64 or height < 64
    except:
        return True
    return False

def initialize(target_class_name, fps):
    global detector
    target_class_index = lookup_class[target_class_name]
    detector = YOLODetector(output_folder=CANDIDATES_DIR, targetClass=target_class_index, buffer_interval=fps)

def generate_frames(video_path):
    return capture.extract_frames(video_path, FRAMES_DIR)

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

def prepare_match_inputs(reference_dir, candidate_dir, output_dir, embed_func, model_type):
    reference_images = sorted(glob(os.path.join(reference_dir, "*.jpg")))
    candidate_images = sorted(glob(os.path.join(candidate_dir, "*.jpg")))

    if not reference_images or not candidate_images:
        print("[WARN] No reference or candidate images found.")
        return None, None, None

    os.makedirs(output_dir, exist_ok=True)

    print("[INFO] Embedding reference images...")
    ref_embeds = []
    for ref in reference_images:
        if is_image_too_small(ref, model_type):
            continue
        emb = embed_func(ref)
        if emb is not None:
            ref_embeds.append((ref, emb))

    return ref_embeds, candidate_images, output_dir

def compare_and_save_matches(ref_embeds, candidate_images, embed_func, prefix, model_type, top_k=10, threshold=0.0):
    all_scores = []
    for cand in candidate_images:
        if is_image_too_small(cand, model_type):
            continue
        print(f"[INFO] [{prefix.upper()}] Comparing candidate: {cand}", flush=True)
        cand_emb = embed_func(cand)
        if cand_emb is None:
            continue
        scores = [CosineSimilarity(dim=1)(cand_emb, ref_emb).item() for _, ref_emb in ref_embeds]
        max_score = max(scores)
        all_scores.append((cand, max_score))

    if threshold > 0:
        all_scores = [match for match in all_scores if match[1] >= threshold]

    if not all_scores:
        print(f"[INFO] No {prefix} matches found above threshold.")
        return False

    all_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = all_scores[:top_k]

    for rank, (img_path, _) in enumerate(top_matches, start=1):
        new_filename = f"{prefix}_{rank:02d}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, new_filename))

    print(f"[INFO] Top {len(top_matches)} {prefix.upper()} matches saved to '{OUTPUT_DIR}/'")
    return True

def find_best_face_matches(target_class_name, top_k=10, fallback_to_dino=True):
    ref_embeds, candidate_images, _ = prepare_match_inputs(
        REFERENCE_CROPS_DIR, CANDIDATES_DIR, OUTPUT_DIR, embed_face, model_type="face"
    )
    if ref_embeds is None:
        return

    success = compare_and_save_matches(ref_embeds, candidate_images, embed_face, prefix="face", model_type="face", top_k=top_k, threshold=0.60)

    if not success and fallback_to_dino:
        print("[INFO] No confident face match found. Falling back to DINO...")
        find_best_dino_matches(top_k=top_k)

def find_best_dino_matches(top_k=10):
    ref_embeds, candidate_images, _ = prepare_match_inputs(
        REFERENCE_CROPS_DIR, CANDIDATES_DIR, OUTPUT_DIR, load_and_embed, model_type="dino"
    )
    if ref_embeds is None:
        return

    compare_and_save_matches(ref_embeds, candidate_images, load_and_embed, prefix="dino", model_type="dino", top_k=top_k)

def find_best_matches(target_class_name, top_k=10):
    if target_class_name == "humans":
        find_best_face_matches(target_class_name, top_k=top_k)
    else:
        find_best_dino_matches(top_k=top_k)

def cleanup():
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    shutil.rmtree(CANDIDATES_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_CROPS_DIR, ignore_errors=True)
    shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
