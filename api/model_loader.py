import json
import sys
from pathlib import Path

import torch

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

CNN_DIR = BASE_DIR / "cnn"
CNN_MODELS_DIR = CNN_DIR / "models"

MODEL_PATH = CNN_MODELS_DIR / "wildlife_cnn.pth"
LABELS_PATH = CNN_DIR / "results" / "species_labels.json"

# Importing CNN architecture directly
sys.path.insert(0, str(CNN_MODELS_DIR))

from CNNModel import SimpleCNN


# Select device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Cache loaded model
_model = None
_class_names = None
_image_size = None


def load_model():
    global _model, _class_names, _image_size

    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained CNN not found: {MODEL_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Species labels not found: {LABELS_PATH}"
        )

    # Load species labels
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    class_names = {
        int(index): species
        for index, species in labels.items()
    }

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_state_dict."
        )

    # Confirm output classes match the supplied labels
    num_classes = len(class_names)

    if set(class_names) != set(range(num_classes)):
        raise ValueError(
            "Species label indices must start at 0 "
            "and be consecutive."
        )

    # Recreate Cnn architecture
    model = SimpleCNN(
        num_classes=num_classes
    )

    # Load trained weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Move model to device
    model.to(DEVICE)

    # Set evaluation mode
    model.eval()

    _model = model
    _class_names = class_names
    _image_size = checkpoint.get("image_size", 128)

    print("CNN loaded successfully!")
    print(f"Device: {DEVICE}")
    print(f"Classes: {num_classes}")
    print(f"Image size: {_image_size}")

    return _model


def get_class_names():
    load_model()
    return _class_names


def get_image_size():
    load_model()
    return _image_size


def get_device():
    return DEVICE