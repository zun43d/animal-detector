import cv2
import math
import cvzone
from ultralytics import YOLO
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Initialize tkinter root window (it won't be shown)
root = Tk()
root.withdraw()  # Hide the root window

# Open file dialog to choose a file
file_path = askopenfilename(title="Select Image or Video", filetypes=[("All Files", "*.*"), ("Image Files", "*.png;*.jpg;*.jpeg"), ("Video Files", "*.mp4;*.avi")])

# Check if a file was selected
if not file_path:
    print("No file selected. Exiting...")
    exit()

# Load YOLO model
model = YOLO("Weights/best.pt")
classNames = ['With Helmet', 'Without Helmet']

# Check if the file exists
if not os.path.exists(file_path):
    print("Error: File not found.")
    exit()

# Check if the input is an image
if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
    print("Processing image...")
    img = cv2.imread(file_path)
    if img is None:
        print("Error: Could not load image.")
        exit()

    # Perform YOLO detection on the image
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

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:  # Assume it's a video
    print("Processing video...")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame. Exiting...")
            break

        # Perform YOLO detection on the frame
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

        # Display the video frame
        cv2.imshow("Video", img)
        # Close window on 'q' or when window is closed manually
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
