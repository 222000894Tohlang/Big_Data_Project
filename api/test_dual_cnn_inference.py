
from pathlib import Path

from inference import predict_species


IMAGE_PATH = (
    Path(__file__).resolve().parent
    / "test_images"
    / "tiger.jpg"
)


def test_model(model_name: str):
    print(f"\nTesting: {model_name}")

    image_bytes = IMAGE_PATH.read_bytes()

    result = predict_species(
        image_bytes,
        model_name=model_name,
    )

    print(f"Model: {result['model_name']}")
    print(f"Species: {result['species']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Top predictions: {result['top_predictions']}")
    print(f"Timing: {result['timing']}")

    assert result["model_name"] == model_name
    assert result["species"]
    assert 0 <= result["confidence"] <= 1
    assert len(result["top_predictions"]) == 3

    print("PASS: Inference successful")


if __name__ == "__main__":
    test_model("cnn")
    test_model("cnn_leakage_aware")

    print("\nBoth CNN inference tests passed.")
