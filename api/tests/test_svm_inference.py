
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from svm_inference import predict_svm_species


image_path = API_DIR / "test_images" / "tiger.jpg"
image_bytes = image_path.read_bytes()

for model_name in ["svm_clean", "svm_original"]:
    print(f"\nTesting: {model_name}")

    result = predict_svm_species(
        image_bytes,
        model_name=model_name,
    )

    print("Predicted species:", result["species"])
    print("Decision score:", result["decision_score"])
    print("Inference time:", result["timing"]["total_ms"], "ms")
