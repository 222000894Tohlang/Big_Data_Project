from pathlib import Path

from preprocessing import preprocess_image


IMAGE_PATH = Path("test_images/zebra.jpeg")


def test_preprocessing():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    image_bytes = IMAGE_PATH.read_bytes()

    tensor = preprocess_image(image_bytes)

    print("Image preprocessing successful!")
    print("Tensor shape:", tensor.shape)
    print("Tensor dtype:", tensor.dtype)
    print("Minimum pixel value:", tensor.min().item())
    print("Maximum pixel value:", tensor.max().item())


if __name__ == "__main__":
    test_preprocessing()