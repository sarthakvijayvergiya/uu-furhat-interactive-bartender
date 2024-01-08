import cv2
import joblib
from feat import Detector
from PIL import Image
import numpy as np
import time
import queue
import threading
# from step4 import BartenderBot, Furhat

# [Your Furhat and BartenderBot class definitions here]

# [Your emotion detection subsystem code here]

# Queue for communication between threads
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
        aus_flat = np.array(aus[0]).flatten()[
            :20
        ]  # Adjust number of features if needed
        return aus_flat
    else:
        return None


# Function to run the emotion detection system
def run_emotion_detection():
    # Initialize the detector
    detector = Detector()

    # Emotion label mapping based on your training data
    # emotion_labels = {
    #     0: "neutral",
    #     1: "happy",
    #     2: "sad",
    #     3: "surprise",
    #     4: "fear",
    #     5: "disgust",
    #     6: "angry"
    # }
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
    model = joblib.load("best_emotion_recognition_model_svm.pkl")

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

            # push emotion to queue
            emotion_queue.put(emotion)

            # Display the result
            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Show the frame
        cv2.imshow("Emotion Recognition", frame)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # [Your emotion detection code here]
    # Inside the while loop, you'll need to add code to communicate detected emotions to the BartenderBot.
    # For example, you could use a shared variable or a queue to pass this information.


# # Function to run the Furhat interaction system
# def run_furhat_interaction():
#     # Initialize Furhat and BartenderBot
#     furhat_robot = Furhat("localhost")
#     furhat_robot.set_voice(name="Matthew")
#     furhat_robot.set_led(red=200, green=50, blue=50)

#     bot = BartenderBot(furhat_robot)

#     while True:
#         if not emotion_queue.empty():
#             # Get the latest detected emotion
#             detected_emotion = emotion_queue.get()
#             print("Detected emotion:", detected_emotion)
#     # user_emotion = "happy"  # This would come from your emotion detection subsystem
#     # user_speech = "It's my birthday today!"  # This would come from speech recognition

#     # bot.on_user_interaction(user_emotion, user_speech)
#     # [Add code here to interact with the BartenderBot based on the detected emotions]
#     # This part of the code will retrieve the detected emotion from the emotion detection system and use it to interact with the user.


# Running both systems in separate threads
emotion_thread = threading.Thread(target=run_emotion_detection)
# furhat_thread = threading.Thread(target=run_furhat_interaction)

emotion_thread.start()
# furhat_thread.start()

# emotion_thread.join()
# furhat_thread.join()
