import torch

from model_loader import (
    load_model,
    get_device,
    get_class_names,
)


def test_final_model():

    print("\nFinal CNN Integration Test\n")

    model = load_model()

    device = torch.device(get_device())

    class_names = get_class_names()

    print(f"Device: {device}")
    print(f"Number of classes: {len(class_names)}")

    # Create a dummy image tensor.
    # This tests the model's output shape,
    # not its classification accuracy.

    dummy_image = torch.zeros(
        (1, 3, 128, 128),
        device=device,
    )

    with torch.inference_mode():
        output = model(dummy_image)

    print(f"Input shape: {tuple(dummy_image.shape)}")
    print(f"Output shape: {tuple(output.shape)}")

    assert output.shape == (
        1,
        len(class_names),
    ), "CNN output shape does not match class count."

    assert torch.isfinite(output).all(), (
        "CNN produced non-finite output values."
    )

    print("\nPASS: CNN output dimensions verified")
    print("PASS: CNN outputs contain finite values")
    print("PASS: Final CNN integration test completed")


if __name__ == "__main__":
    test_final_model()