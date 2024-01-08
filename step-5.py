import threading
import cv2
import joblib
from feat import Detector
from PIL import Image
import numpy as np
import time
from furhat_remote_api import FurhatRemoteAPI

# [Your Furhat and BartenderBot class definitions here]

# [Your emotion detection subsystem code here]

# Function to run the emotion detection system
def run_emotion_detection():
    # [Your emotion detection code here]
    # Inside the while loop, you'll need to add code to communicate detected emotions to the BartenderBot.
    # For example, you could use a shared variable or a queue to pass this information.

# Function to run the Furhat interaction system
def run_furhat_interaction():
    # Initialize Furhat and BartenderBot
    furhat = FurhatRemoteAPI("localhost")
    bot = BartenderBot(furhat)

    # [Add code here to interact with the BartenderBot based on the detected emotions]
    # This part of the code will retrieve the detected emotion from the emotion detection system and use it to interact with the user.

# Running both systems in separate threads
emotion_thread = threading.Thread(target=run_emotion_detection)
furhat_thread = threading.Thread(target=run_furhat_interaction)

emotion_thread.start()
furhat_thread.start()

emotion_thread.join()
furhat_thread.join()
