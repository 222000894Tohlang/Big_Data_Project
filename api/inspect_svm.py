from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "traditional-ml" / "models"

files = [
    MODEL_DIR / "clean_split_sklearn.joblib",
    MODEL_DIR / "other_split_sklearn.joblib",
]

for filename in files:
    print(f"\n--- {filename} ---")

    try:
        model = joblib.load(filename)

        print("\nModel configuration:")
        print(model["config"])

        print("\nSpecies:")
        print(model["species_list"])

        print("\nExpected HOG features:")
        print(model["scaler"].n_features_in_)

        print("\nSVM classes:")
        print(model["svm"].classes_)
        
        print("Object type:", type(model))

        if isinstance(model, dict):
            print("Dictionary keys:", list(model.keys()))

            for key, value in model.items():
                print(f"{key}: {type(value)}")

        if hasattr(model, "classes_"):
            print("Classes:", model.classes_)

        if hasattr(model, "n_features_in_"):
            print("Number of features:", model.n_features_in_)

        if hasattr(model, "steps"):
            print("Pipeline steps:", model.steps)

        if hasattr(model, "kernel"):
            print("SVM kernel:", model.kernel)

    except Exception as error:
        print("Error:", error)
