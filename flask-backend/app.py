from flask import Flask, request, Response
from flask_cors import CORS
import os
import shutil
import main

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024
UPLOAD_FOLDER = 'input'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process():
    if request.method != 'POST':
        return "Method not allowed", 405
    video = request.files.get('video')
    reference_images = request.files.getlist('reference_images')
    target = request.form.get('target', 'cars')

    if not video or not reference_images:
        return "Missing video or reference images", 400

    video_path = os.path.join(UPLOAD_FOLDER, 'input.mp4')
    video.save(video_path)

    ref_dir = "reference"
    shutil.rmtree(ref_dir, ignore_errors=True)
    os.makedirs(ref_dir, exist_ok=True)
    for i, img in enumerate(reference_images):
        img.save(os.path.join(ref_dir, f"ref_{i}.jpg"))

    def generate_logs():
        yield "[INFO] Extracting frames from video...\n"
        main.generate_frames(video_path)
        yield "[INFO] Running YOLO on extracted frames...\n"
        main.process_all_frames_sequentially(target)

        yield "[INFO] Running DINO similarity matching...\n"

        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        main.find_best_matches(top_k=3)

        sys.stdout = old_stdout
        yield mystdout.getvalue()

        yield "[INFO] Process complete.\n"

    return Response(generate_logs(), mimetype='text/plain')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False) 
