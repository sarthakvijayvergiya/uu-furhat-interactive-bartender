import cv2
import joblib
from feat import Detector
from PIL import Image
import numpy as np
import time

# Initialize the detector
detector = Detector()

# Emotion label mapping based on your training data
emotion_labels = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "angry"
}

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


# Load the trained model
model = joblib.load('./../../models/best_emotion_recognition_model.pkl')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set desired FPS
desired_fps = 10  # Adjust as needed
frame_interval = 1 / desired_fps

last_time = time.time()

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

        # Display the result
        cv2.putText(frame, f'Emotion: {emotion}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
