import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_and_display_data():
    data = pd.read_csv("processed/aus.csv")  # Load your dataset
    class_distribution = data["expression"].value_counts()
    print("Class distribution:")
    print(class_distribution)
    return data

def split_data(data):
    labels = data["expression"]
    features = data.drop(["file", "face", "expression"], axis=1)  # Exclude non-feature columns

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Create a pipeline that first imputes missing values, then fits the model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # You can change strategy to 'mean' or 'most_frequent'
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    test_predictions = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy for {type(model).__name__}: {test_accuracy:.2f}")
    return test_accuracy

def main():
    data = load_and_display_data()
    X_train, X_test, y_train, y_test = split_data(data)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier()
    }

    best_accuracy = 0
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        print(f"Training and evaluating {name}")
        accuracy = train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(f"The best model is {best_model_name} with an accuracy of {best_accuracy:.2f}")

    # Save the best performing model
    if best_model is not None:
        joblib.dump(best_model, 'best_emotion_recognition_model.pkl')
        print(f"Saved the best model ({best_model_name}) to 'best_emotion_recognition_model.pkl'.")

if __name__ == "__main__":
    main()

