import io
import time
from pathlib import Path

import numpy as np
import requests

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
)

from results_exporter import export_results

API_URL = "http://127.0.0.1:8000"

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

IMAGE_PATH = TEST_IMAGES_DIR / "tiger.jpg"


def image_to_bytes(image, quality=95):
    """
    Convert a PIL image into JPEG bytes.
    """

    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    return buffer.getvalue()


def add_noise(image, noise_level=20):
    """
    Add random noise to an image.
    """

    image_array = np.array(
        image.convert("RGB"),
        dtype=np.float32,
    )

    rng = np.random.default_rng(42)

    noise = rng.normal(
        loc=0,
        scale=noise_level,
        size=image_array.shape,
    )

    noisy_array = np.clip(
        image_array + noise,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(noisy_array)


def create_image_variations(image):
    """
    Generate different image quality variations.
    """

    variations = {
        "original": image_to_bytes(image),

        "low_quality_jpeg": image_to_bytes(
            image,
            quality=20,
        ),

        "medium_quality_jpeg": image_to_bytes(
            image,
            quality=50,
        ),

        "blurred": image_to_bytes(
            image.filter(
                ImageFilter.GaussianBlur(radius=3)
            )
        ),

        "darkened": image_to_bytes(
            ImageEnhance.Brightness(image).enhance(0.4)
        ),

        "brightened": image_to_bytes(
            ImageEnhance.Brightness(image).enhance(1.6)
        ),

        "grayscale": image_to_bytes(
            image.convert("L").convert("RGB")
        ),

        "noisy": image_to_bytes(
            add_noise(image)
        ),
    }

    return variations


def test_image_quality(name, image_bytes):
    """
    Send image to API and record prediction results.
    """

    start_time = time.perf_counter()

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                f"{name}.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
        timeout=60,
    )

    response_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.raise_for_status()

    prediction = response.json()

    return {
        "variation": name,
        "species": prediction["species"],
        "confidence": prediction["confidence"],
        "response_time_ms": round(
            response_time_ms,
            2,
        ),
        "inference_ms": prediction["timing"]["inference_ms"],
    }


def run_quality_tests():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")

    variations = create_image_variations(image)

    print("\nWildlife Image Quality Evaluation\n")

    results = []

    for name, image_bytes in variations.items():

        result = test_image_quality(
            name,
            image_bytes,
        )
        results.append(result)

        print(f"Variation: {result['variation']}")

        print(
            f"Prediction: {result['species']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.2%}"
        )

        print(
            f"CNN inference: "
            f"{result['inference_ms']:.2f} ms"
        )

        print(
            f"HTTP response: "
            f"{result['response_time_ms']:.2f} ms"
        )

        print("-" * 40)

    export_results(
        "image_quality_results.csv",
        results,
    )


if __name__ == "__main__":
    run_quality_tests()