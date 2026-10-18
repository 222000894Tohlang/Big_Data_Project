
from model_loader import MODEL_PATHS
from svm_loader import MODEL_DIR, MODEL_FILES, load_svm_model


def svm_is_available(model_name: str) -> bool:
    """Check that an SVM bundle exists and can be loaded."""
    path = MODEL_DIR / MODEL_FILES[model_name]

    if not path.is_file():
        return False

    try:
        load_svm_model(model_name)
        return True
    except Exception:
        return False


def get_available_models():
    original_available = MODEL_PATHS["cnn"].is_file()
    leakage_aware_available = (
        MODEL_PATHS["cnn_leakage_aware"].is_file()
    )

    clean_available = svm_is_available("svm_clean")
    other_available = svm_is_available("svm_original")

    return {
        "cnn": {
            "name": "Original CNN",
            "available": original_available,
            "status": (
                "ready" if original_available
                else "checkpoint_missing"
            ),
            "classes": 22,
            "image_size": 128,
        },
        "cnn_leakage_aware": {
            "name": "Leakage-Aware CNN",
            "available": leakage_aware_available,
            "status": (
                "ready" if leakage_aware_available
                else "checkpoint_missing"
            ),
            "classes": 22,
            "image_size": 128,
        },
        "svm": {
            "name": "HOG + Support Vector Machine",
            "available": clean_available,
            "status": (
                "ready" if clean_available
                else "integration_pending"
            ),
            "classes": 22,
            "image_size": 224,
        },
        "svm_clean": {
            "name": "Clean-Split HOG + SVM",
            "available": clean_available,
            "status": (
                "ready" if clean_available
                else "integration_pending"
            ),
            "classes": 22,
            "image_size": 224,
        },
        "svm_original": {
            "name": "Other-Split HOG + SVM",
            "available": other_available,
            "status": (
                "ready" if other_available
                else "integration_pending"
            ),
            "classes": 22,
            "image_size": 224,
        },
    }
