
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from svm_loader import load_svm_model, get_svm_species


for model_name in ["svm_clean", "svm_original"]:
    print(f"\nTesting: {model_name}")

    bundle = load_svm_model(model_name)

    print("Classifier:", type(bundle["svm"]).__name__)
    print("Scaler:", type(bundle["scaler"]).__name__)
    print("Features:", bundle["svm"].n_features_in_)
    print("Species count:", len(get_svm_species(model_name)))
    print("Status: Loaded successfully")
