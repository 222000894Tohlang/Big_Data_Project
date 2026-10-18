
import torch

from model_loader import (
    get_class_names,
    get_image_size,
    load_model,
)


def test_checkpoint(model_name):
    print(f"\nTesting: {model_name}")

    model = load_model(model_name)

    class_names = get_class_names(model_name)
    image_size = get_image_size(model_name)

    device = next(model.parameters()).device

    dummy_input = torch.zeros(
        1, 3, image_size, image_size,
        device=device,
    )

    with torch.inference_mode():
        output = model(dummy_input)

    print(f"Input: {tuple(dummy_input.shape)}")
    print(f"Output: {tuple(output.shape)}")
    print(f"Classes: {len(class_names)}")

    assert output.shape == (1, 22)
    assert torch.isfinite(output).all()

    print("PASS: Checkpoint loaded successfully")


if __name__ == "__main__":
    test_checkpoint("cnn")
    test_checkpoint("cnn_leakage_aware")

    print("\nBoth CNN checkpoint tests passed.")
