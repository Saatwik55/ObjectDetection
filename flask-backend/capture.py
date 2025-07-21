import cv2
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def extract_frames(video_path, output_folder='frames'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {video_path}")
        return -1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Video FPS: {video_fps}, Total frames: {total_frames}")

    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = os.path.join(output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        frame_idx += 1

    cap.release()
    logging.info(f"Extracted {saved_count} frames to '{output_folder}'")
    return int(video_fps)
