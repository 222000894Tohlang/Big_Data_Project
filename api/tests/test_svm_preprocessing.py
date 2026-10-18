
<<<<<<< HEAD
import sys
=======
>>>>>>> data
from pathlib import Path

import numpy as np

<<<<<<< HEAD
API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from svm_preprocessing import preprocess_svm_image


def test_image_preprocessing():
    # Use a sample wildlife image from the project.
    image_path = API_DIR / "test_images" / "tiger.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    features = preprocess_svm_image(image_bytes)

    assert features.shape == (1, 6084)
    assert np.isfinite(features).all()

    print("Image:", image_path.name)
    print("Feature shape:", features.shape)
    print("All values finite:", np.isfinite(features).all())
    print("SVM preprocessing successful.")


if __name__ == "__main__":
    test_image_preprocessing()
=======
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
>>>>>>> data
