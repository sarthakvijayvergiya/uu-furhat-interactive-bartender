import cv2
import joblib
from feat import Detector
from PIL import Image
import numpy as np
from flask import Flask, Response, render_template
import time
import threading
import queue
from step4 import BartenderBot, Furhat

app = Flask(__name__)

# Initialize the detector
detector = Detector()

# Emotion label mapping based on your training data
emotion_labels = {
    0: "happy",
    1: "happy",
    2: "sad",
    3: "happy",
    4: "sad",
    5: "sad",
    6: "sad",
}

# Load the trained model
model = joblib.load('best_emotion_recognition_model_svm.pkl')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set desired FPS
desired_fps = 30  # Adjust as needed
frame_interval = 15 / desired_fps

# Create a queue for communication between threads
emotion_queue = queue.Queue()

def extract_features(frame, detector):
    # Convert frame to format expected by py-feat (PIL Image)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Detect faces and extract features
    detections = detector.detect_faces(frame_pil)
    if len(detections) == 0:
        return None

    detected_landmarks = detector.detect_landmarks(frame, detections)

    # Assuming that the model was trained using the features from the first detected face
    if len(detected_landmarks) > 0:
        aus = detector.detect_aus(frame_pil, detected_landmarks)
    else:
        return None

    # Check if AUs are extracted and handle the structure
    if isinstance(aus, list) and len(aus) > 0:
        # Flatten the structure if it's a nested list or array
        # and ensure only the expected number of features are returned
        aus_flat = np.array(aus[0]).flatten()[:20]  # Adjust number of features if needed
        return aus_flat
    else:
        return None

def generate_frames():

    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    last_time = time.time()  # Initialize last_time here

    while True:
        # Control frame rate
        current_time = time.time()
        if current_time - last_time < frame_interval:
            continue
        last_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame and extract features
        features = extract_features(frame, detector)
        print("Features:", features)

        # Check if features are extracted
        if features is not None and len(features) > 0:
            # Predict emotion
            emotion = model.predict([features])[0]
            emotion = emotion_labels.get(emotion, "Unknown")

            # Draw the result on the frame
            cv2.putText(frame, f'Emotion: {emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Put the detected emotion in the queue for furhat processing
            emotion_queue.put(emotion)

        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_furhat_interaction():
    # Initialize Furhat and BartenderBot
    furhat_robot = Furhat("localhost")
    furhat_robot.set_voice(name="Matthew")
    furhat_robot.set_led(red=200, green=50, blue=50)

    bot = BartenderBot(furhat_robot)

    while True:
        if not emotion_queue.empty():
            # Get the latest detected emotion
            detected_emotion = emotion_queue.get()
            print("Detected emotion:", detected_emotion)
            bot.on_user_interaction(detected_emotion, "")
            # Your Furhat interaction logic goes here

# Start Flask app in a separate thread
t_app = threading.Thread(target=app.run, args=('0.0.0.0', 5000), daemon=True)
t_app.start()

# Start the emotion processing thread
t_emotion = threading.Thread(target=generate_frames)
t_emotion.start()

# Start the Furhat interaction thread
t_furhat = threading.Thread(target=run_furhat_interaction)
t_furhat.start()

# Wait for the threads to finish
t_app.join()
t_emotion.join()
t_furhat.join()

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
