
import json
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# CONSTANT PATHS AND PARAMETERS
DATASET_ROOT = Path(r"D:\Studies\Dataset\archive (1)")

# Prefer final_clean_split.csv for the final experiment.
SPLIT_CSV = Path(
    r"D:\Studies\Dataset\archive (1)\splits\leakage_aware_split.csv"
)

SPECIES_LABELS_PATH = (
    DATASET_ROOT / "results" / "species_labels.json"
)

MODEL_DIR = DATASET_ROOT / "models"
MODEL_PATH = MODEL_DIR / "wildlife_cnn_leakage_aware.pth"
METRICS_PATH = MODEL_DIR / "leakage_aware_training_metrics.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001
PATIENCE = 5
NUM_WORKERS = 0
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CNN MODEL
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

# IMAGE PATH RESOLUTION
def resolve_image_path(path_value):

    raw_path = str(path_value).strip()
    path = Path(raw_path)
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            DATASET_ROOT / path,
            DATASET_ROOT / "archive" / path,
            DATASET_ROOT / "archive" / "images" / path,
            DATASET_ROOT / "images" / path,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Image not found: {raw_path}\n"
        f"Checked these locations:\n"
        + "\n".join(str(p) for p in candidates)
    )

# LOAD SPECIES LABELS
def load_species_mapping(json_path):

    with open(json_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict):
        raise ValueError(
            "species_labels.json must contain a JSON object."
        )

    # Determine the mapping direction.
    if all(str(key).isdigit() for key in mapping.keys()):
        species_to_idx = {
            str(species): int(index)
            for index, species in mapping.items()
        }
    else:
        species_to_idx = {
            str(species): int(index)
            for species, index in mapping.items()
        }

    if len(set(species_to_idx.values())) != len(species_to_idx):
        raise ValueError(
            "Species labels contain duplicate numeric class IDs."
        )

    return species_to_idx

# DATASET
class WildlifeDataset(Dataset):
    def __init__(self, dataframe, species_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = row["resolved_path"]
        species = str(row["species"])

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from exc

        label = self.species_to_idx[species]

        return image, label

# Transforms
def create_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, evaluation_transform

# PREPARE AND VALIDATE SPLITS
def normalize_split_name(value):
    value = str(value).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }

    return aliases.get(value, value)

def prepare_data():
    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}\n"
            "Check SPLIT_CSV at the top of this script."
        )

    if not SPECIES_LABELS_PATH.is_file():
        raise FileNotFoundError(
            f"Species mapping not found: {SPECIES_LABELS_PATH}"
        )

    df = pd.read_csv(SPLIT_CSV)

    required_columns = {
        "path",
        "species",
        "identity",
        "leakage_aware_split",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Split CSV is missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(
        subset=[
            "path",
            "species",
            "identity",
            "leakage_aware_split",
        ]
    ).copy()

    df["species"] = df["species"].astype(str).str.strip()
    df["identity"] = df["identity"].astype(str).str.strip()

    df["split"] = df["leakage_aware_split"].apply(
        normalize_split_name
    )

    allowed_splits = {"train", "validation", "test"}
    unexpected = set(df["split"].unique()) - allowed_splits

    if unexpected:
        raise ValueError(
            f"Unexpected split names: {unexpected}\n"
            "Expected train, validation, and test."
        )

    species_to_idx = load_species_mapping(SPECIES_LABELS_PATH)

    unknown_species = (
        set(df["species"].unique()) - set(species_to_idx.keys())
    )

    if unknown_species:
        raise ValueError(
            "Species found in the CSV but missing from "
            f"species_labels.json: {sorted(unknown_species)}"
        )

    # Confirm each identity is assigned to exactly one split.
    identity_split_counts = df.groupby("identity")["split"].nunique()

    conflicting_identities = identity_split_counts[
        identity_split_counts > 1
    ]

    if not conflicting_identities.empty:
        raise ValueError(
            "Identity leakage detected: some identities occur "
            "in more than one split. Examples: "
            f"{conflicting_identities.index[:10].tolist()}"
        )

    # Resolve and verify image paths.
    print("\nResolving image paths...")

    resolved_paths = []

    for path in tqdm(
        df["path"].tolist(),
        desc="Checking images",
        unit="image",
        dynamic_ncols=True,
    ):
        resolved_paths.append(str(resolve_image_path(path)))

    df["resolved_path"] = resolved_paths

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty:
        raise ValueError("The training split is empty.")

    if val_df.empty:
        raise ValueError(
            "The validation split is empty. Check how the split CSV "
            "was generated before training."
        )

    if test_df.empty:
        raise ValueError(
            "The test split is empty. Check how the split CSV "
            "was generated before training."
        )

    # Explicit overlap checks.
    train_ids = set(train_df["identity"])
    val_ids = set(val_df["identity"])
    test_ids = set(test_df["identity"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\nDataset split summary")
    print("-" * 55)

    for split_name, split_df in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{split_name:12s}: "
            f"{len(split_df):,} images | "
            f"{split_df['identity'].nunique():,} identities | "
            f"{split_df['species'].nunique():,} species"
        )

    print("\nIdentity overlap check: PASSED")
    print(f"Number of species classes: {len(species_to_idx)}")
    print(f"Device: {DEVICE}")

    return train_df, val_df, test_df, species_to_idx

# CREATE DATA LOADERS
def create_data_loaders(
    train_df,
    val_df,
    test_df,
    species_to_idx,
):
    train_transform, eval_transform = create_transforms()

    train_dataset = WildlifeDataset(
        train_df,
        species_to_idx,
        train_transform,
    )

    val_dataset = WildlifeDataset(
        val_df,
        species_to_idx,
        eval_transform,
    )

    test_dataset = WildlifeDataset(
        test_df,
        species_to_idx,
        eval_transform,
    )

    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

# LOAD TRAINED MODEL
def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    phase="Training",
    epoch=None,
    total_epochs=None,
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    start_time = time.time()

    if epoch is not None and total_epochs is not None:
        description = (
            f"Epoch {epoch}/{total_epochs} - {phase}"
        )
    else:
        description = phase

    progress = tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size

        running_loss = total_loss / total_images
        running_accuracy = total_correct / total_images

        elapsed = time.time() - start_time

        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            accuracy=f"{running_accuracy * 100:.2f}%",
            images=f"{total_images:,}/{len(loader.dataset):,}",
            elapsed=f"{elapsed / 60:.1f}m",
        )

    if total_images == 0:
        raise RuntimeError(f"{phase} processed zero images.")

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy

# MAIN TRAINING LOOP
def train_model():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    (
        train_df,
        val_df,
        test_df,
        species_to_idx,
    ) = prepare_data()

    train_loader, val_loader, test_loader = create_data_loaders(
        train_df,
        val_df,
        test_df,
        species_to_idx,
    )

    model = SimpleCNN(
        num_classes=len(species_to_idx)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\n" + "=" * 65)
    print("STARTING LEAKAGE-AWARE CNN TRAINING")
    print("=" * 65)
    print(f"Model: SimpleCNN")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Training images: {len(train_df):,}")
    print(f"Validation images: {len(val_df):,}")
    print(f"Test images: {len(test_df):,}")
    print(f"Checkpoint: {MODEL_PATH}")

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print("\n" + "-" * 65)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 65)

        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            phase="Training",
            epoch=epoch,
            total_epochs=EPOCHS,
        )

        val_loss, val_accuracy = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            phase="Validation",
            epoch=epoch,
            total_epochs=EPOCHS,
        )

        epoch_seconds = time.time() - epoch_start

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "species_to_idx": species_to_idx,
                "idx_to_species": {
                    index: species
                    for species, index in species_to_idx.items()
                },
                "num_classes": len(species_to_idx),
                "image_size": IMAGE_SIZE,
                "architecture": "SimpleCNN",
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_split_csv": str(SPLIT_CSV),
                "random_seed": RANDOM_SEED,
            }

            torch.save(checkpoint, MODEL_PATH)

            print(f"\nBest model saved: {MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "epoch_seconds": epoch_seconds,
            "best_model_updated": improved,
        })

        pd.DataFrame(history).to_csv(
            METRICS_PATH,
            index=False,
        )

        print("\nEpoch summary")
        print(f"  Training loss:       {train_loss:.4f}")
        print(f"  Training accuracy:   {train_accuracy * 100:.2f}%")
        print(f"  Validation loss:     {val_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy * 100:.2f}%")
        print(f"  Epoch duration:      {epoch_seconds / 60:.2f} minutes")
        print(f"  Best validation loss:{best_val_loss:10.4f}")
        print(f"  Best epoch:          {best_epoch}")
        print(
            f"  No improvement:      "
            f"{epochs_without_improvement}/{PATIENCE} epochs"
        )

        if epochs_without_improvement >= PATIENCE:
            print(
                "\nEarly stopping activated: validation loss "
                "has not improved."
            )
            break

    total_training_seconds = time.time() - training_start

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(
        f"Total training time: "
        f"{total_training_seconds / 60:.2f} minutes"
    )
    print(f"Metrics CSV: {METRICS_PATH}")

    # LOAD BEST CHECKPOINT BEFORE FINAL TEST EVALUATION
    print("\nLoading the best checkpoint for final test evaluation...")

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the test set once after model selection.
    test_loss, test_accuracy = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        phase="Final Test Evaluation",
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Best epoch:    {best_epoch}")
    print(f"Saved model:   {MODEL_PATH}")

    # Append final test metrics to a separate file.
    test_metrics_path = (
        MODEL_DIR / "leakage_aware_test_metrics.csv"
    )

    pd.DataFrame([{
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
        "test_images": len(test_df),
        "test_identities": test_df["identity"].nunique(),
        "model_path": str(MODEL_PATH),
        "split_csv": str(SPLIT_CSV),
    }]).to_csv(test_metrics_path, index=False)

    print(f"Test metrics saved: {test_metrics_path}")

# ENTRY POINT
if __name__ == "__main__":
    train_model()
