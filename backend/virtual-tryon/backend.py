from flask import Flask, request, send_file, jsonify
import os
import uuid
import subprocess
import glob
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/test'
IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'image')
CLOTH_FOLDER = os.path.join(UPLOAD_FOLDER, 'cloth')
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(CLOTH_FOLDER, exist_ok=True)

@app.route('/api/tryon', methods=['POST'])
def tryon():
    try:
        filename = request.args.get('filename', 'output.png')
        filename2 = request.args.get('filename2','output.png')

        print(f"dddddddddddddd: {filename}")
        print(f"Received pair: {filename} {filename2}")

        if not filename or not filename2:
            return jsonify({"error": "Missing filename or filename2"}), 400
        with open('data/test_pairs.txt', 'w') as f:
            f.write(f"{filename}.jpg {filename2}.jpg")
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        person_filename = f"person_{session_id}.jpg"
        cloth_filename = f"cloth_{session_id}.jpg"

        # Get uploaded files
        person_img = request.files.get('person_image')
        cloth_img = request.files.get('clothes_image')

        if not person_img or not cloth_img:
            return jsonify({"error": "Missing image files"}), 400

        # Save uploaded images
        person_path = os.path.join(IMAGE_FOLDER, person_filename)
        cloth_path = os.path.join(CLOTH_FOLDER, cloth_filename)
        person_img.save(person_path)
        cloth_img.save(cloth_path)
        print(f"Saving person image to {person_path}")
        print(f"Saving cloth image to {cloth_path}")

        # Update test_pairs.txt
        with open('test_pairs.txt', 'w') as f:
            f.write(f"{person_filename} {cloth_filename}\n")

        # Create session-specific result directory
        session_result_dir = os.path.join('result', session_id)
        os.makedirs(session_result_dir, exist_ok=True)
        print(f"Session result directory: {session_result_dir}")
        # Run test.py with TOM stage using subprocess
        result = subprocess.run([
            'python', 'test.py',
            '--name', session_id,
            '--stage', 'TOM',
            '--datamode', 'test',
            '--data_list', 'test_pairs.txt',
            '--checkpoint', 'checkpoints/tom_train_new/tom_final.pth',
            '--result_dir', session_result_dir,
            '--workers', '1'
        ], capture_output=True, text=True)

        # Debug log
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            return jsonify({"error": f"Model failed to process images. {result.stderr}"}), 500

        # Find generated try-on image
      
        # RESULTS_DIR = rf'C:\Users\naimb\OneDrive\Desktop\vfr-20250429T104034Z-1-001\cp-vton\result\{session_id}\tom_final.pth\test\try-on'
        # tryon_path = os.path.join(RESULTS_DIR, f"{filename}")
        # print(f"Result images found: {result_images}")
        # print(f"Tryon path: {tryon_path}")
        
        base_results_dir = os.path.join(
        'C:', os.sep, 'Users', 'naimb', 'OneDrive', 'Desktop',
        'vfr-20250429T104034Z-1-001', 'cp-vton', 'result'
    )

        tryon_path = os.path.join(
            base_results_dir, session_id, 'tom_final.pth', 'test', 'try-on'
        )
        # Full path to the expected result image
        result_image_path = os.path.join(tryon_path, filename)+".png"
        print(f"Looking for result image at: {result_image_path}")

        return send_file(result_image_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
