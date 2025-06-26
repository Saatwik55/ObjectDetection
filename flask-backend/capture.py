import cv2
import os
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calc_sharpness(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    sharpness = variance_of_laplacian(small_frame)
    return sharpness

def extract_frames(video_path, output_folder='frames', extract_fps=10, best_n_per_sec=5, max_workers=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    logging.info(f"Video FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration_sec:.2f}s")

    frame_interval = int(round(video_fps / extract_fps))
    if frame_interval == 0:
        frame_interval = 1

    frames_per_second = defaultdict(list)

    frame_idx = 0
    frames_to_process = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            sec = int(frame_idx / video_fps)
            frames_to_process.append((sec, frame_idx, frame))
        frame_idx += 1
    cap.release()

    #Calculate sharpness in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        sharpness_results = list(executor.map(lambda x: (x[0], x[1], x[2], calc_sharpness(x[2])), frames_to_process))

    for sec, idx, frame, sharpness in sharpness_results:
        frames_per_second[sec].append((sharpness, idx, frame))
    saved_count = 0
    for sec, frames_info in frames_per_second.items():
        frames_info.sort(key=lambda x: x[0], reverse=True)
        best_frames = frames_info[:best_n_per_sec]

        for sharpness, idx, frame in best_frames:
            filename = os.path.join(output_folder, f"frame_{idx:06d}_sec{sec}_sharp{int(sharpness)}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

    logging.info(f"Extracted {saved_count} sharp frames to '{output_folder}'")