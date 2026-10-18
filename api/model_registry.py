
from pathlib import Path
from model_loader import MODEL_PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CNN_CHECKPOINT = (
    PROJECT_ROOT / "cnn" / "models" / "wildlife_cnn.pth"
)

SVM_CHECKPOINT = (
    PROJECT_ROOT
    / "traditional-ml"
    / "models"
    / "svm_model_sklearn.joblib"
)


def get_available_models():
    original_available = MODEL_PATHS["cnn"].is_file()
    leakage_aware_available = (
        MODEL_PATHS["cnn_leakage_aware"].is_file()
    )

    return {
        "cnn": {
            "name": "Original CNN",
            "available": original_available,
            "status": (
                "ready"
                if original_available
                else "checkpoint_missing"
            ),
            "classes": 22,
            "image_size": 128,
        },
        "cnn_leakage_aware": {
            "name": "Leakage-Aware CNN",
            "available": leakage_aware_available,
            "status": (
                "ready"
                if leakage_aware_available
                else "checkpoint_missing"
            ),
            "classes": 22,
            "image_size": 128,
        },
        "svm": {
            "name": "HOG + Support Vector Machine",
            "available": False,
            "status": "integration_pending",
            "classes": 22,
            "image_size": 224,
        },
    }
