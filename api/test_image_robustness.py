import io
import time
from pathlib import Path
from results_exporter import export_results
import requests
from PIL import Image


API_URL = "http://127.0.0.1:8000"
TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

IMAGE_SIZES = [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

IMAGE_FORMATS = [
    ("JPEG", "image/jpeg", "jpg"),
    ("PNG", "image/png", "png"),
]


def create_test_image(
    original_image: Image.Image,
    size: tuple[int, int],
    image_format: str,
) -> bytes:

    resized_image = original_image.resize(size)

    buffer = io.BytesIO()

    resized_image.save(
        buffer,
        format=image_format,
    )

    return buffer.getvalue()


def test_image(
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> dict:

    start_time = time.perf_counter()

    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                filename,
                image_bytes,
                content_type,
            )
        },
        timeout=60,
    )

    response_time_ms = (time.perf_counter() - start_time) * 1000

    result = {
        "filename": filename,
        "file_size_kb": round(len(image_bytes) / 1024, 2),
        "status_code": response.status_code,
        "response_time_ms": round(response_time_ms, 2),
    }

    if response.status_code == 200:

        prediction = response.json()

        result.update(
            {
                "species": prediction["species"],
                "confidence": prediction["confidence"],
                "preprocessing_ms": prediction["timing"]["preprocessing_ms"],
                "inference_ms": prediction["timing"]["inference_ms"],
                "processing_ms": prediction["timing"]["total_ms"],
            }
        )

    else:

        result["error"] = response.text

    return result


def run_robustness_tests():

    image_path = TEST_IMAGES_DIR / "tiger.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    with Image.open(image_path) as image:
        original_image = image.convert("RGB")

    print("\nWildlife Image Robustness Evaluation\n")

    results = []

    for width, height in IMAGE_SIZES:

        for image_format, content_type, extension in IMAGE_FORMATS:

            image_bytes = create_test_image(
                original_image,
                (width, height),
                image_format,
            )

            filename = f"tiger_{width}x{height}.{extension}"

            result = test_image(
                image_bytes,
                filename,
                content_type,
            )
            result["width"] = width
            result["height"] = height
            result["format"] = image_format

            results.append(result)

            print(f"Image: {filename}")
            print(f"Size: {result['file_size_kb']} KB")
            print(f"HTTP Status: {result['status_code']}")
            print(f"Response Time: " f"{result['response_time_ms']} ms")

            if result["status_code"] == 200:

                print(f"Prediction: {result['species']}")
                print(f"Confidence: " f"{result['confidence']:.2%}")

            else:

                print(f"Error: {result['error']}")

            print("-" * 40)
            
    export_results(
        "image_robustness_results.csv",
        results,
    )


if __name__ == "__main__":
    run_robustness_tests()
