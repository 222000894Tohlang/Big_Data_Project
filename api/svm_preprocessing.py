
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


IMAGE_SIZE = (224, 224)


def extract_hog_features(image_bytes: bytes) -> np.ndarray:
    """
    Convert an uploaded image into a HOG feature vector.
    """

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.reshape(1, -1)
