from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import datetime as dt

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('video.html')

def detect_objects():
    # Load the YOLOv8 model with .pt weights
    model = YOLO('model/febrian.pt')

    # Open video file
    cap = cv2.VideoCapture('/Users/hesda/Documents/febria/Videos/cars.mp4')

    try:
        while True:
            # Read frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Perform inference on the image
            results = model(frame)

            # Get detection results
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy()

            # Draw bounding boxes and labels on the frame
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(pred_classes[i])]} {pred_scores[i]:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Print debug information
                print(f'Detected {label} at [{x1}, {y1}, {x2}, {y2}] with score {pred_scores[i]}')


            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)

            if not ret:
                continue

            # Yield the frame as a byte array
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        # Release the video capture
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
