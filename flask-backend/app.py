from flask import Flask, request, Response, send_from_directory, jsonify
from flask_cors import CORS
import os
import shutil
import main

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 1024
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process():
    main.cleanup()
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
        fps = main.generate_frames(video_path)
        main.initialize(target, fps)
        yield "[INFO] Running YOLO on reference frames...\n"
        main.process_reference_images()
        yield "[INFO] Running YOLO on extracted frames...\n"
        main.process_all_frames()

        yield "[INFO] Running DINO similarity matching...\n"

        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        main.find_best_matches(target)

        sys.stdout = old_stdout
        yield mystdout.getvalue()

        yield "[INFO] Process complete.\n"

    return Response(generate_logs(), mimetype='text/plain')


@app.route('/results', methods=['GET'])
def get_output_images():
    if not os.path.exists(OUTPUT_FOLDER):
        return jsonify([])

    files = sorted([
        f for f in os.listdir(OUTPUT_FOLDER)
        if os.path.isfile(os.path.join(OUTPUT_FOLDER, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    return jsonify(files)


@app.route('/output/<filename>')
def serve_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
