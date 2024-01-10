import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

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
    features = data.drop(["file", "face", "expression", "valence", "arousal"], axis=1)

    # Split into training and test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)

    # Further split the training set into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='weighted')
    val_recall = recall_score(y_val, val_predictions, average='weighted')
    val_f1 = f1_score(y_val, val_predictions, average='weighted')
    
    # Print or log these metrics
    print(f"Validation Accuracy for {type(model).__name__}: {val_accuracy:.2f}")
    print(f"Validation Precision for {type(model).__name__}: {val_precision:.2f}")
    print(f"Validation Recall for {type(model).__name__}: {val_recall:.2f}")
    print(f"Validation F1 Score for {type(model).__name__}: {val_f1:.2f}")
    return val_accuracy

def tune_hyperparameters(X_train, y_train):
    """ Perform hyperparameter tuning for an SVM model """
    param_grid = [
        {"model__kernel": ["poly"], "model__degree": [3, 15, 25, 50]},
        {"model__kernel": ["rbf", "linear", "sigmoid"]}
    ]
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', SVC())
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Optimal parameters for SVC:", grid_search.best_params_)
    return grid_search

def tune_random_forest(X_train, y_train):
    # Define parameter grid
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        # Add other parameters to test
    }

    # Create a pipeline with imputer, scaler, and random forest
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),  # Optional: include if feature scaling is beneficial
        ('model', RandomForestClassifier(random_state=42))
    ])

    # Create randomized search object
    randomized_search = RandomizedSearchCV(pipeline, param_grid, n_iter=100, cv=5, random_state=42)
    randomized_search.fit(X_train, y_train)

    # Best parameters and model
    print("Best Parameters:", randomized_search.best_params_)
    return randomized_search


def main():
    # data = load_and_display_data()
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    # X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    # models = {
    #     "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    #     "KNeighborsClassifier": KNeighborsClassifier()
    # }

    # best_accuracy = 0
    # best_model_name = ""
    # best_model = None

    # for name, model in models.items():
    #     print(f"Training and evaluating {name}")
    #     accuracy = train_and_evaluate(model, X_train_scaled, y_train, X_val_scaled, y_val)
    #     if accuracy > best_accuracy:
    #         best_accuracy = accuracy
    #         best_model = model
    #         best_model_name = name

    # print(f"The best non-SVM model is {best_model_name} with a validation accuracy of {best_accuracy:.2f}")

    # # Save the best performing non-SVM model
    # if best_model is not None:
    #     joblib.dump(best_model, 'best_emotion_recognition_model_non_svm.pkl')
    #     print(f"Saved the best non-SVM model ({best_model_name}) to 'best_emotion_recognition_model_non_svm.pkl'.")

    # # SVM model tuning and evaluation
    # svm_model = tune_hyperparameters(X_train_scaled, y_train).best_estimator_
    # svm_accuracy = train_and_evaluate(svm_model, X_train_scaled, y_train, X_val_scaled, y_val)
    # print(f"Validation Set Accuracy for SVM: {svm_accuracy:.2f}")

    # # Save the tuned SVM model
    # joblib.dump(svm_model, 'best_emotion_recognition_model_svm.pkl')
    # print("Saved the tuned SVM model to 'best_emotion_recognition_model_svm.pkl'.")

    data = load_and_display_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)
    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    models = {
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "SVM": tune_hyperparameters(X_train_scaled, y_train).best_estimator_
    }

    accuracies = {}  # Dictionary to store accuracies for plotting

    for name, model in models.items():
        print(f"Training and evaluating {name}")
        accuracy = train_and_evaluate(model, X_train_scaled, y_train, X_val_scaled, y_val)
        accuracies[name] = accuracy

    # Plotting the accuracies
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Model')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')

    # Save the plot as an image file
    plt.savefig('accuracy_plot.png')
    plt.show()

    best_accuracy = 0
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        print(f"Training and evaluating {name}")
        accuracy = train_and_evaluate(model, X_train_scaled, y_train, X_val_scaled, y_val)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(f"The best model is {best_model_name} with a validation accuracy of {best_accuracy:.2f}")

    # Save the best performing model
    if best_model is not None:
        joblib.dump(best_model, f'best_emotion_recognition_model_{best_model_name.lower()}.pkl')
        print(f"Saved the best model ({best_model_name}) to 'best_emotion_recognition_model_{best_model_name.lower()}.pkl'.")

if __name__ == "__main__":
    main()