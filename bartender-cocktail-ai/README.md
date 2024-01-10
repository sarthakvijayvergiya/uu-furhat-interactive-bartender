# Bartender Cocktail AI

## Introduction
This project aims to build a Bartender Cocktail AI using a combination of emotion detection and Furhat interaction. The system detects users' emotions using facial features and provides socially adaptive responses through Furhat.

## Project Structure
- **/data:** Directory to store datasets.
  - `diffusionfer_dataset.csv`: Your DiffusionFER dataset.

- **/models:** Directory to store trained models.
  - `best_model.pkl`: The best-trained model.

- **/results:** Directory to store any evaluation or result files.
  - `evaluation_results.txt`: Text file to store evaluation metrics.

- **/src:** Source code directory.
  - **emotion_detection:** Subdirectory for emotion detection subsystem.
    - `Step1_data_preprocessing.py`: Handles data preprocessing and AU extraction.
    - `Step2_model_training_evaluation.py`: Generates, trains, evaluates the best model.
    - `Step3_live_emotion_recognition.py`: Runs the model with webcam for live emotion recognition.

  - **bartender_interaction:** Subdirectory for bartender interaction subsystem.
    - `Step4_bartender_furhat.py`: Defines Bartender and Furhat classes.
    - `Step5_integration.py`: Integrates both subsystems.

- `main.py`: Main script to execute the entire project.

- `requirements.txt`: File containing the necessary dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the main script: `python main.py`

## Results
- Describe the key results and insights obtained from the project.
- Include any visualizations or metrics that showcase the system's performance.

## Notes
- Any additional notes or considerations about the project.

## Contributors
- List of contributors and their contributions.

## License
This project is licensed under the [Your License] License - see the [LICENSE.md](LICENSE.md) file for details.
