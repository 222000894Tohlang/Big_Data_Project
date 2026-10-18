from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "traditional-ml" / "models"

MODEL_FILES = {
    "svm_clean": "clean_split_sklearn.joblib",
    "svm_original": "other_split_sklearn.joblib",
}


@lru_cache(maxsize=2)
def load_svm_model(model_name: str = "svm_clean") -> dict:
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown SVM model: {model_name}")

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(f"SVM model file not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError("SVM model bundle must be a dictionary.")

    required_keys = {
        "scaler",
        "svm",
        "species_list",
        "config",
    }

    missing = required_keys - bundle.keys()

    if missing:
        raise ValueError(f"Missing SVM model components: {sorted(missing)}")

    if not isinstance(bundle["scaler"], StandardScaler):
        raise TypeError("Expected a fitted StandardScaler.")

    if not isinstance(bundle["svm"], LinearSVC):
        raise TypeError("Expected a fitted LinearSVC.")

    if bundle["scaler"].n_features_in_ != 6084:
        raise ValueError("Unexpected SVM feature dimension.")

    if bundle["svm"].n_features_in_ != 6084:
        raise ValueError("Unexpected classifier feature dimension.")

    if len(bundle["species_list"]) != 22:
        raise ValueError("Expected 22 species labels.")

    return bundle


def get_svm_species(model_name: str = "svm_clean") -> list[str]:
    bundle = load_svm_model(model_name)

    return [str(species) for species in bundle["species_list"]]
