
<<<<<<< HEAD
from time import perf_counter

import numpy as np

from svm_loader import load_svm_model
from svm_preprocessing import preprocess_svm_image


def predict_svm_species(
    image_bytes: bytes,
    model_name: str = "svm_clean",
) -> dict:
    """
    Predict wildlife species using the trained HOG + LinearSVC model.
    """

    start = perf_counter()

    # 1. Load the trained model bundle.
    bundle = load_svm_model(model_name)

    scaler = bundle["scaler"]
    svm = bundle["svm"]
    species_list = bundle["species_list"]

    # 2. Extract HOG features from the image.
    features = preprocess_svm_image(image_bytes)

    # 3. Apply the fitted training scaler.
    scaled_features = scaler.transform(features)

    # 4. Predict the species class.
    predicted_class = int(svm.predict(scaled_features)[0])

    # 5. Map the class index to its species label.
    species = str(species_list[predicted_class])

    # 6. Obtain the SVM decision score.
    decision_scores = svm.decision_function(scaled_features)
    class_position = np.where(svm.classes_ == predicted_class)[0]

    decision_score = (
        float(decision_scores[0, class_position[0]])
        if len(class_position)
        else None
    )

    elapsed_ms = (perf_counter() - start) * 1000

    return {
        "model": "svm",
        "model_name": model_name,
        "species": species,
        "decision_score": decision_score,
        "timing": {
            "total_ms": round(elapsed_ms, 2),
        },
=======
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
>>>>>>> data
    }
