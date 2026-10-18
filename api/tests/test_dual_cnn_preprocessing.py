
from pathlib import Path

import torch

from preprocessing import preprocess_image
from model_loader import get_image_size


IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "test_images"
    / "tiger.jpg"
)


def test_preprocessing(model_name: str):
    image_bytes = IMAGE_PATH.read_bytes()

    tensor = preprocess_image(
        image_bytes,
        model_name=model_name,
    )

    size = get_image_size(model_name)

    assert tensor.shape == (1, 3, size, size)
    assert torch.isfinite(tensor).all()

    print(f"\nModel: {model_name}")
    print(f"Tensor shape: {tuple(tensor.shape)}")
    print(f"Image size: {size}")
    print("PASS: Preprocessing successful")


if __name__ == "__main__":
    test_preprocessing("cnn")
    test_preprocessing("cnn_leakage_aware")

    print("\nBoth preprocessing tests passed.")
