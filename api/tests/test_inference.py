from pathlib import Path

from inference import predict_species


IMAGE_PATH = Path("test_images/zebra.jpeg")


def test_inference():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    image_bytes = IMAGE_PATH.read_bytes()

    prediction = predict_species(image_bytes)

    print("\nCNN Prediction Result")
    print("----------------------")

    print("Species:", prediction["species"])
    print("Confidence:", prediction["confidence"])

    print("----------------------")


if __name__ == "__main__":
    test_inference()