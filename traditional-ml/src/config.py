from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data source (switch here to change splits)------------------
CSV_PATH = PROJECT_ROOT / "splits" / "final_clean_split.csv"
SPLIT_COL = "leakage_aware_split"
PATH_COL = "path"
LABEL_COL = "species"
DATASET_ROOT = Path(r"D:\PythonProject\traditional-ml")

# Image settings ----------------------------------------------
IMG_SIZE = (224, 224) # (width, height). try (128, 128). later

# HOG parameters ----------------------------------------------
HOG_ORIENTATIONS = 9
HOG_PPC = (16, 16) # pixels per cell
HOG_CPB = (2, 2) # cells per block
HOG_BLOCK_NORM = "L2-Hys"

# Artifacts ---------------------------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Tag used in artifact filenames so multiple experiments don't overwrite each other
RUN_TAG = f"{CSV_PATH.stem}_{SPLIT_COL}_{IMG_SIZE[0]}"
FEATURES_PATH = ARTIFACTS_DIR / f"hog_{RUN_TAG}.npz"
MODEL_PATH = ARTIFACTS_DIR / f"svm_{RUN_TAG}.joblib"