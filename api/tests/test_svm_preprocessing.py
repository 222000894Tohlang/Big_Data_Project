
from pathlib import Path

import numpy as np

from svm_preprocessing import extract_hog_features


BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "test_images" / "tiger.jpg"


def test_hog_preprocessing():

    print("\nHOG Preprocessing Test\n")

    image_bytes = IMAGE_PATH.read_bytes()

    features = extract_hog_features(image_bytes)

    print(f"Image: {IMAGE_PATH.name}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature count: {features.shape[1]}")
    print(f"Data type: {features.dtype}")

    assert features.ndim == 2
    assert features.shape[0] == 1
    assert np.isfinite(features).all()

    print("\nPASS: HOG features extracted successfully")
    print("PASS: All feature values are finite")


if __name__ == "__main__":
    test_hog_preprocessing()
