from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import subprocess

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model with custom weights
model = YOLO("Weights/best-100.pt")  # Update the path to your custom weights
classNames = ['Bear', 'Cheetah', 'Crocodile', 'Elephant', 'Fox', 'Giraffe', 'Hedgehog', 'Human', 'Leopard', 'Lion', 'Lynx', 'Ostrich', 'Rhinoceros', 'Tiger', 'Zebra'] 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save uploaded file temporarily
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the image with YOLO
    img = cv2.imread(file_path)
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Save the result image to send back to the user
    result_filename = 'result_' + file.filename
    result_path = os.path.join(UPLOAD_FOLDER, result_filename)
    cv2.imwrite(result_path, img)
    
    return jsonify({"result_image": f'/uploads/{result_filename}'})

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save uploaded video temporarily
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the video with YOLO
    cap = cv2.VideoCapture(file_path)
    temp_output_path = os.path.join(UPLOAD_FOLDER, 'temp_processed_' + file.filename)
    output_filename = 'processed_' + os.path.splitext(file.filename)[0] + '.mp4'
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Temporary codec
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process the image with YOLO
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        # Write the processed frame to the temporary video
        out.write(img)

    # Release everything once done
    cap.release()
    out.release()

    # Re-encode the video with FFmpeg
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", temp_output_path,  # Input file
        "-c:v", "libx264",  # Use H.264 codec
        "-preset", "fast",  # Set encoding speed/quality tradeoff
        "-crf", "23",  # Control output quality (lower is better)
        output_path
    ]
    
    subprocess.run(ffmpeg_command, check=True)

    # Remove the temporary video
    os.remove(temp_output_path)

    return jsonify({"result_video": f'/uploads/{output_filename}'})

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
