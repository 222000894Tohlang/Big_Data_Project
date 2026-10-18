import csv
import statistics
import time

from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000"

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "test_images" / "tiger.jpg"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_FILE = RESULTS_DIR / "performance_results.csv"

WARMUP_REQUESTS = 5
TEST_REQUESTS = 50


def send_prediction(session, image_bytes):
    """
    Send an image to the prediction API and
    measure the complete HTTP response time.
    """

    start_time = time.perf_counter()

    response = session.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "tiger.jpg",
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

    result = response.json()

    return {
        "species": result["species"],
        "confidence": result["confidence"],
        "preprocessing_ms": result["timing"]["preprocessing_ms"],
        "inference_ms": result["timing"]["inference_ms"],
        "processing_ms": result["timing"]["total_ms"],
        "response_time_ms": round(response_time_ms, 2),
    }


def calculate_percentile(values, percentile):
    """
    Calculate percentile using linear interpolation.
    """

    sorted_values = sorted(values)

    position = (
        len(sorted_values) - 1
    ) * percentile / 100

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)

    fraction = position - lower

    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def print_statistics(name, values):
    """
    Print descriptive statistics.
    """

    print(f"\n{name}")

    print(f"Average: {statistics.mean(values):.2f} ms")
    print(f"Median: {statistics.median(values):.2f} ms")
    print(f"Minimum: {min(values):.2f} ms")
    print(f"Maximum: {max(values):.2f} ms")

    print(
        f"Standard deviation: "
        f"{statistics.stdev(values):.2f} ms"
    )

    print(
        f"95th percentile: "
        f"{calculate_percentile(values, 95):.2f} ms"
    )


def run_performance_test():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_bytes = IMAGE_PATH.read_bytes()

    print("\nWildlife CNN Performance Evaluation")

    print(f"\nWarm-up requests: {WARMUP_REQUESTS}")
    print(f"Measured requests: {TEST_REQUESTS}")

    results = []

    with requests.Session() as session:

        # Warm-up requests are not included
        # in the final statistics.

        print("\nWarming up CNN...")

        for _ in range(WARMUP_REQUESTS):
            send_prediction(session, image_bytes)

        print("Warm-up complete.")

        print("\nRunning performance tests...\n")

        for request_number in range(1, TEST_REQUESTS + 1):

            result = send_prediction(
                session,
                image_bytes,
            )

            result["request_number"] = request_number

            results.append(result)

            print(
                f"Request {request_number:02d}: "
                f"{result['response_time_ms']:.2f} ms"
            )

    # Save results

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "request_number",
                "species",
                "confidence",
                "preprocessing_ms",
                "inference_ms",
                "processing_ms",
                "response_time_ms",
            ],
        )

        writer.writeheader()

        writer.writerows(results)

    # Calculate statistics

    response_times = [
        result["response_time_ms"]
        for result in results
    ]

    inference_times = [
        result["inference_ms"]
        for result in results
    ]

    processing_times = [
        result["processing_ms"]
        for result in results
    ]

    print_statistics(
        "HTTP Response Time",
        response_times,
    )

    print_statistics(
        "CNN Inference Time",
        inference_times,
    )

    print_statistics(
        "Total Processing Time",
        processing_times,
    )

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_performance_test()