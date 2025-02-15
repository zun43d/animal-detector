import cv2
import math
import cvzone
from ultralytics import YOLO
import os

# Initialize video capture
video_path = "C:/Users/arifu/Downloads/Helmet-Project/Helmet-Project/Media/vide2.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
classNames = ['With Helmet', 'Without Helmet']

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Get video properties for resolution and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the processed video
output_path = os.path.join(UPLOAD_FOLDER, 'processed_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for .mp4 file
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

    # Write the processed frame to the output video
    out.write(img)

    # Display the processed frame
    cv2.imshow("Processed Video", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()

# Return the path to the saved processed video
print(f"Processed video saved to: {output_path}")
