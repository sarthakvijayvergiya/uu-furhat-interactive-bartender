import os
import pandas as pd
from feat import Detector

# Initialize the detector
detector = Detector(device="cpu")

# Paths to dataset and output directories
dataset_dir = "./DiffusionFER"
output_dir = "./processed"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Path to the CSV file with valence, arousal, and expression
csv_path = "./DiffusionFER/DiffusionEmotion_S/dataset_sheet.csv"

# Load the dataset sheet
dataset_sheet = pd.read_csv(csv_path)

aus_data = []

def replace_second_underscore(string):
    parts = string.split("_", 2)  # Split into 3 parts, splitting at most 2 underscores
    if len(parts) == 3:
        return parts[0] + "_" + parts[1] + "/" + parts[2]
    else:
        return string  # Return the original string if there are not enough underscores
    
# Iterate over each row in the dataset sheet
for index, row in dataset_sheet.iterrows():
    # Construct the image path
    image_relative_path = replace_second_underscore(row['subDirectory_filePath'])
    image_path = os.path.join(dataset_dir, image_relative_path)

    print(image_path)

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}. Skipping.")
        continue

    # Detect faces and extract features from the image
    detections = detector.detect_image(image_path)
    if detections.empty:
        print(f"No faces detected in {image_path}. Skipping.")
        continue

    # Extract Action Units and other relevant features
    for i, detection in detections.iterrows():
        aus = {col: detection[col] for col in detection.index if col.startswith('AU')}
        aus_data.append({
            "file": os.path.basename(image_path).split('.')[0],
            "face": i,
            "valence": row['valence'],
            "arousal": row['arousal'],
            "expression": row['expression'],
            **aus
        })

# Create a DataFrame from the extracted data
aus_df = pd.DataFrame(aus_data)

# Save the extracted features to a CSV file
aus_csv_path = os.path.join(output_dir, "aus.csv")
aus_df.to_csv(aus_csv_path, index=False)

print("Processing complete. AU data saved.")
