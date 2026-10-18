
import time

import torch

from model_loader import (
    load_model,
    get_class_names,
    get_device,
)

from preprocessing import preprocess_image


def predict_species(
    image_bytes: bytes,
    model_name: str = "cnn",
) -> dict:
    """
    Run inference using the selected CNN model
    and measure processing time.
    """

    model = load_model(model_name)
    device = get_device()
    class_names = get_class_names(model_name)

    total_start = time.perf_counter()

    # 1. Preprocessing
    preprocessing_start = time.perf_counter()

    image_tensor = preprocess_image(
        image_bytes,
        model_name=model_name,
    )

    image_tensor = image_tensor.to(device)

    preprocessing_time_ms = (
        time.perf_counter() - preprocessing_start
    ) * 1000

    # 2. CNN inference
    inference_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(image_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

    inference_time_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    # 3. Postprocessing
    probabilities = torch.softmax(
        outputs,
        dim=1,
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(3, len(class_names)),
        dim=1,
    )

    top_predictions = []

    for index, probability in zip(
        top_indices[0].tolist(),
        top_probabilities[0].tolist(),
    ):
        top_predictions.append({
            "species": class_names[index],
            "confidence": round(probability, 4),
        })

    best_prediction = top_predictions[0]

    total_time_ms = (
        time.perf_counter() - total_start
    ) * 1000

    return {
        "model_name": model_name,
        "species": best_prediction["species"],
        "confidence": best_prediction["confidence"],
        "top_predictions": top_predictions,
        "timing": {
            "preprocessing_ms": round(
                preprocessing_time_ms, 2
            ),
            "inference_ms": round(
                inference_time_ms, 2
            ),
            "total_ms": round(
                total_time_ms, 2
            ),
        },
    }
