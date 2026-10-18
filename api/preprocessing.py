
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model_loader import get_image_size


def validate_image(image_bytes: bytes) -> Image.Image:
    """
    Validate and decode an uploaded image.

    Returns a PIL image in RGB format.
    """

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()

            if image.format not in ("JPEG", "PNG"):
                raise ValueError(
                    "Only JPEG and PNG images are supported."
                )

            return image.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(
            "The uploaded file is not a valid image."
        ) from exc


def preprocess_image( image_bytes: bytes,
    model_name: str = "cnn",) -> torch.Tensor:
    """
    Convert uploaded image bytes into a CNN-ready tensor.

    Output shape: [1, 3, height, width]
    """

    image = validate_image(image_bytes)

    image_size = get_image_size(model_name)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
