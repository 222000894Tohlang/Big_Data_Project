
from svm_loader import (
    SVMUnavailableError,
    load_svm_model,
    svm_model_exists,
)

from svm_preprocessing import extract_hog_features


def predict_svm(image_bytes: bytes) -> dict:
    """
    Predict wildlife species using HOG + SVM.
    """

    if not svm_model_exists():
        raise SVMUnavailableError(
            "SVM prediction is unavailable. "
            "A CPU-compatible model is required."
        )

    model = load_svm_model()

    features = extract_hog_features(image_bytes)

    prediction = model.predict(features)

    return {
        "model": "svm",
        "species": str(prediction[0]),
    }
