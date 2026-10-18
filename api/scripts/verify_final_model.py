import json
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "cnn" / "models" / "wildlife_cnn.pth"
)

LABELS_PATH = (
    BASE_DIR / "cnn" / "results" / "species_labels.json"
)


def verify_model():

    print("\nWildlife CNN Model Verification\n")

    # Check model file

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"Model: {MODEL_PATH.name}")

    print(
        f"Model size: "
        f"{MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )

    # Load checkpoint metadata
    # Only load checkpoints from trusted teammates.

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Expected a dictionary-based CNN checkpoint."
        )

    print("\nCheckpoint keys:")

    for key in checkpoint.keys():
        print(f"  - {key}")

    # Check class mapping

    species_to_id = checkpoint.get("species_to_id")

    if species_to_id is not None:

        print(
            f"\nCheckpoint classes: "
            f"{len(species_to_id)}"
        )

        print("\nCheckpoint species mapping:")

        for species, class_id in sorted(
            species_to_id.items(),
            key=lambda item: item[1],
        ):
            print(f"  {class_id}: {species}")

    # Check image size

    image_size = checkpoint.get("image_size")

    print(f"\nTraining image size: {image_size}")

    # Check external label mapping

    if LABELS_PATH.exists():

        with open(
            LABELS_PATH,
            encoding="utf-8",
        ) as file:

            labels = json.load(file)

        print(
            f"\nExternal label entries: "
            f"{len(labels)}"
        )

        print("\nExternal labels:")

        print(
            json.dumps(
                labels,
                indent=2,
            )
        )

    else:

        print("\nExternal label mapping not found.")

    print("\nModel metadata inspection complete.")


if __name__ == "__main__":
    verify_model()