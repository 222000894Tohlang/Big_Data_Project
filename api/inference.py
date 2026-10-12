import torch

import time

from model_loader import (
    load_model,
    get_class_names,
    get_device,
)

from preprocessing import preprocess_image


def predict_species(image_bytes: bytes) -> dict:
    """
    Run CNN inference and measure processing time.
    """

    model = load_model()
    device = get_device()
    class_names = get_class_names()

    total_start = time.perf_counter()

    # 1. Measure preprocessing time
    preprocessing_start = time.perf_counter()

    image_tensor = preprocess_image(image_bytes)
    image_tensor = image_tensor.to(device)

    preprocessing_time_ms = (time.perf_counter() - preprocessing_start) * 1000

    # 2. Measure CNN inference time
    inference_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(image_tensor)

        # Synchronise GPU before stopping timer
        if device.type == "cuda":
            torch.cuda.synchronize()

    inference_time_ms = (time.perf_counter() - inference_start) * 1000

    # 3. Postprocessing
    probabilities = torch.softmax(outputs, dim=1)

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1,
    )

    top_predictions = []

    for index, probability in zip(
        top_indices[0].tolist(),
        top_probabilities[0].tolist(),
    ):
        top_predictions.append(
            {
                "species": class_names[index],
                "confidence": round(probability, 4),
            }
        )

    best_prediction = top_predictions[0]

    total_time_ms = (time.perf_counter() - total_start) * 1000

    return {
        "species": best_prediction["species"],
        "confidence": best_prediction["confidence"],
        "top_predictions": top_predictions,
        "timing": {
            "preprocessing_ms": round(preprocessing_time_ms, 2),
            "inference_ms": round(inference_time_ms, 2),
            "total_ms": round(total_time_ms, 2),
        },
    }
