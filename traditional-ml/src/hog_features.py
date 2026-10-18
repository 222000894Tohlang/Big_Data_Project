from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from config import IMG_SIZE,DATASET_ROOT, HOG_ORIENTATIONS, HOG_PPC, HOG_CPB, HOG_BLOCK_NORM

# Load an image, convert to RGB, resize, and convert to grayscale
def load_image_gray(path, size=IMG_SIZE):
    """Load → RGB → resize → grayscale."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img) 
    return rgb2gray(arr) 

# Extract HOG features from a grayscale image
def extract_hog(gray):
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PPC,
        cells_per_block=HOG_CPB,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )

# Compute the length of the HOG feature vector for a given image size
def hog_feature_dim(size=IMG_SIZE):
    """Compute HOG vector length without loading an image."""
    h, w = size[1], size[0]
    cy = h // HOG_PPC[0] - HOG_CPB[0] + 1
    cx = w // HOG_PPC[1] - HOG_CPB[1] + 1
    return cx * cy * HOG_CPB[0] * HOG_CPB[1] * HOG_ORIENTATIONS

# Helper function to resolve relative paths to absolute paths based on the dataset root
def _resolve(path):
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(DATASET_ROOT) / p

# Load an image, convert to RGB, resize, and convert to grayscale
def load_image_gray(path, size=IMG_SIZE):
    img = Image.open(_resolve(path)).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img)
    return rgb2gray(arr)