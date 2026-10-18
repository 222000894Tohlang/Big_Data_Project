
import sys
from pathlib import Path

import numpy as np

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
