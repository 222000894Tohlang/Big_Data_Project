import os
import numpy as np
import sklearn

from src.config import FEATURES_PATH

data = np.load(FEATURES_PATH, allow_pickle=True)

print("CPU logical threads:", os.cpu_count())
print("scikit-learn version:", sklearn.__version__)
print("Training shape:", data["X_train"].shape)
print("Validation shape:", data["X_val"].shape)
print("Test shape:", data["X_test"].shape)
print("Feature dtype:", data["X_train"].dtype)

print(
    "Training array size (GB):",
    round(data["X_train"].nbytes / 1e9, 2)
)