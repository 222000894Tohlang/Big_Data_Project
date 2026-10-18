
import json
import sys
from pathlib import Path

import torch

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

CNN_DIR = BASE_DIR / "cnn"
CNN_MODELS_DIR = CNN_DIR / "models"

LABELS_PATH = CNN_DIR / "results" / "species_labels.json"

MODEL_PATHS = {
    "cnn": CNN_MODELS_DIR / "wildlife_cnn.pth",
    "cnn_leakage_aware": (
        CNN_MODELS_DIR / "wildlife_cnn_leakage_aware.pth"
    ),
}

# Preserve the original constant for compatibility
MODEL_PATH = MODEL_PATHS["cnn"]

sys.path.insert(0, str(CNN_MODELS_DIR))

from CNNModel import SimpleCNN


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Cache models independently
_model_cache = {}
_class_names_cache = {}
_image_size_cache = {}


def _validate_model_name(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown CNN model: {model_name}. "
            f"Available models: {list(MODEL_PATHS)}"
        )


def _read_class_names():
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Species labels not found: {LABELS_PATH}"
        )

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    class_names = {
        int(index): str(species)
        for index, species in labels.items()
    }

    num_classes = len(class_names)

    if set(class_names) != set(range(num_classes)):
        raise ValueError(
            "Species label indices must start at 0 "
            "and be consecutive."
        )

    return class_names


def _validate_checkpoint_labels(checkpoint, class_names):
    """
    Confirm the checkpoint mapping matches
    the species_labels.json mapping.
    """

    species_to_id = checkpoint.get("species_to_id")

    if species_to_id is None:
        species_to_id = checkpoint.get("species_to_idx")

    if species_to_id is not None:
        expected = {
            species: index
            for index, species in class_names.items()
        }

        actual = {
            str(species): int(index)
            for species, index in species_to_id.items()
        }

        if actual != expected:
            raise ValueError(
                "Checkpoint species mapping does not "
                "match species_labels.json."
            )

    idx_to_species = checkpoint.get("idx_to_species")

    if idx_to_species is not None:
        actual = {
            int(index): str(species)
            for index, species in idx_to_species.items()
        }

        if actual != class_names:
            raise ValueError(
                "Checkpoint idx_to_species mapping "
                "does not match species_labels.json."
            )


def load_model(model_name: str = "cnn"):
    """
    Load the requested CNN checkpoint.

    Defaults to the original CNN to preserve
    existing API behaviour.
    """

    _validate_model_name(model_name)

    if model_name in _model_cache:
        return _model_cache[model_name]

    model_path = MODEL_PATHS[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"CNN checkpoint not found: {model_path}"
        )

    class_names = _read_class_names()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_state_dict."
        )

    _validate_checkpoint_labels(
        checkpoint,
        class_names,
    )

    num_classes = len(class_names)

    model = SimpleCNN(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    image_size = int(
        checkpoint.get("image_size", 128)
    )

    if image_size <= 0:
        raise ValueError(
            "Checkpoint image_size must be positive."
        )

    _model_cache[model_name] = model
    _class_names_cache[model_name] = class_names
    _image_size_cache[model_name] = image_size

    print(f"\nCNN loaded: {model_name}")
    print(f"Checkpoint: {model_path.name}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {num_classes}")
    print(f"Image size: {image_size}")

    return model


def get_class_names(model_name: str = "cnn"):
    load_model(model_name)
    return _class_names_cache[model_name]


def get_image_size(model_name: str = "cnn"):
    load_model(model_name)
    return _image_size_cache[model_name]


def get_device():
    return DEVICE
