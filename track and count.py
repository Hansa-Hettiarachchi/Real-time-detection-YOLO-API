from flask import Flask, request,Response
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import cv2
import threading
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Track and Count System"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load the YOLO face detection model
model = YOLO(os.path.relpath("best.pt"))  # Ensure this model is suitable for face detection
# model.cuda()
model.fuse()

# Global variables for the real-time frame and lock
frame = None
frame_lock = threading.Lock()

# Line zone parameters
LINE_START = sv.Point(1000, 1000)
LINE_END = sv.Point(1500, 200)
line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(
    thickness=4, 
    text_thickness=4, 
    text_scale=2
)
box_annotator = sv.BoxAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2
)

in_count = 0
out_count = 0

# Flag to control video capture loop
video_capture_active = False

# Function to start video capture from the webcam
def start_video_capture():
    global frame, video_capture_active
    video_capture_active = True
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution if needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set the desired FPS
    
    while cap.isOpened() and video_capture_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Pass the frame to the model for processing
        processed_frame = process_frame(frame)

        # Sleep to maintain the target processing frame rate
        time.sleep(0.1)
    
    cap.release()
    video_capture_active = False

# Function to process the frame using the YOLO face detection model
def process_frame(frame):
    results = model.track(frame, tracker='bytetrack.yaml', show=False, agnostic_nms=True, persist=True)
    # results = model(frame)
    # detections = sv.Detections.from_yolov8(results)
    
    for result in results:
        detections = sv.Detections.from_yolov8(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )
        in_count = line_counter.in_count
        out_count = line_counter.out_count
        # Print detection classes to console
        for label in labels:
            print(label)
            print(in_count)
            print(out_count)
    return frame, in_count, out_count

@app.route('/submit', methods=['POST'])
def submit():
    data = request.json
    if 'id' in data:
        if data['id'] == 'start_capture':
            threading.Thread(target=start_video_capture, daemon=True).start()
            return "Video capture initiated"
        elif data['id'] == 'stop_capture':
            global video_capture_active
            video_capture_active = False
            return "Video capture stopped"
    return "Invalid request"

# # Function to generate frames for video streaming
# def generate_frames():
#     global frame
#     while True:
#         with frame_lock:
#             if frame is not None:
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Start the video capture in a separate thread
# threading.Thread(target=start_video_capture, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)
