
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog


IMAGE_SIZE = (224, 224)
EXPECTED_FEATURES = 6084


def preprocess_svm_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert an uploaded image into HOG features.

    Returns:
        NumPy array with shape (1, 6084).
    """
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            # Convert to grayscale for HOG extraction.
            image = image.convert("L")

            # Resize to the model's expected image dimensions.
            image = image.resize(
                IMAGE_SIZE,
                resample=Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(image, dtype=np.uint8)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "Unable to read the uploaded image."
        ) from exc

    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    if features.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} HOG features, "
            f"received {features.shape[0]}."
        )

    return features.reshape(1, -1)
