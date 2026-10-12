import time
from pathlib import Path

import requests
from results_exporter import export_results

API_URL = "http://127.0.0.1:8000"

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"


def measure_response_time(image_path: Path):
    """
    Measure total HTTP response time for a prediction.
    """

    with open(image_path, "rb") as image_file:

        start_time = time.perf_counter()

        response = requests.post(
            f"{API_URL}/predict",
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=60,
        )

        end_time = time.perf_counter()

    response.raise_for_status()

    response_time_ms = (
        end_time - start_time
    ) * 1000

    result = response.json()

    return {
        "image": image_path.name,
        "species": result["species"],
        "confidence": result["confidence"],
        "preprocessing_ms": result["timing"]["preprocessing_ms"],
        "inference_ms": result["timing"]["inference_ms"],
        "processing_ms": result["timing"]["total_ms"],
        "response_time_ms": round(response_time_ms, 2),
    }


if __name__ == "__main__":

    print("\nWildlife API Response Time Evaluation\n")

    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))

    if not images:
        raise RuntimeError(
            "No JPEG test images found in test_images."
        )

    results = []

    for image_path in images:

        result = measure_response_time(image_path)

        results.append(result)

        print(f"Image: {result['image']}")
        print(f"Species: {result['species']}")
        print(f"HTTP response: {result['response_time_ms']:.2f} ms")
        print("-" * 40)

    export_results(
        "response_time_results.csv",
        results,
    )