
import numpy as np
from skimage.feature import hog

# Match the image dimensions used during training.
image = np.zeros((224, 224), dtype=np.uint8)

features = hog(
    image,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    feature_vector=True,
)

print("HOG feature shape:", features.shape)
print("Expected features:", 6084)

assert features.shape == (6084,), (
    f"Expected 6084 features, got {features.shape[0]}"
)

print("HOG feature dimension verified.")
