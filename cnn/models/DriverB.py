
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    r"D:\Studies\Dataset\archive (1)\models\wildlife_cnn_leakage_aware.pth"
)

IMAGE_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. CNN ARCHITECTURE
# Must match TrainLeakageAware.py exactly
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 3. IMAGE TRANSFORM
# Must match validation/test preprocessing
# ============================================================

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

def load_model(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}\n"
            "Train the model first or update MODEL_PATH."
        )

    try:
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

    required_keys = {
        "model_state_dict",
        "num_classes",
        "species_to_idx",
    }

    missing = required_keys - set(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint is missing required fields: {missing}"
        )

    species_to_idx = checkpoint["species_to_idx"]

    # Convert saved class IDs to integers.
    species_to_idx = {
        str(species): int(index)
        for species, index in species_to_idx.items()
    }

    model = SimpleCNN(
        num_classes=int(checkpoint["num_classes"])
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx_to_species = {
        index: species
        for species, index in species_to_idx.items()
    }

    print("Model loaded successfully.")
    print(f"Device: {DEVICE}")
    print(f"Species classes: {len(idx_to_species)}")

    return model, idx_to_species


# ============================================================
# 5. PREDICT IMAGE
# ============================================================

def predict_image(model, idx_to_species, image_path, top_k=5):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # Load the original image for display.
    with Image.open(image_path) as source:
        original_image = source.convert("RGB")

    transform = get_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    number_of_classes = probabilities.numel()
    top_k = min(top_k, number_of_classes)

    scores, indices = torch.topk(probabilities, k=top_k)

    results = []

    print("\n" + "=" * 55)
    print("WILDLIFE SPECIES CLASSIFICATION")
    print("=" * 55)
    print(f"Image: {image_path}")
    print(f"Device: {DEVICE}")

    for rank, (score, index) in enumerate(
        zip(scores.tolist(), indices.tolist()),
        start=1,
    ):
        species = idx_to_species.get(
            index,
            f"Unknown class {index}",
        )

        confidence = score * 100
        results.append((species, confidence))

        print(
            f"{rank}. {species:<20} "
            f"Confidence: {confidence:.2f}%"
        )

    predicted_species, confidence = results[0]

    print("-" * 55)
    print(f"Predicted species: {predicted_species}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 55)

    # Display image and prediction results.
    plt.figure(figsize=(10, 7))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title(
        f"Predicted: {predicted_species}\n"
        f"Confidence: {confidence:.2f}%",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    return results


# ============================================================
# 6. DRIVER
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test the trained wildlife CNN on an image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Full path to the wildlife image to classify.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Optional path to a trained model checkpoint.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top species predictions to display.",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    model_path = Path(args.model)

    model, idx_to_species = load_model(model_path)

    predict_image(
        model=model,
        idx_to_species=idx_to_species,
        image_path="D:\\Studies\\Dataset\\archive (1)\\data\\dog\\0.0_f94e984431525aad.jpg",
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
