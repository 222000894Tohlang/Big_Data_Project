from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# CONSTANTS
DATASET_DIR = Path("data/Dataset")
INPUT_FILE = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = Path("splits")
OUTPUT_FILE = OUTPUT_DIR / "final_original_split.csv"

RANDOM_SEED = 42
VALIDATION_RATIO = 0.15

# LOAD METADATA
metadata = pd.read_csv(INPUT_FILE, low_memory=False)

required_columns = [
    "identity",
    "path",
    "species",
    "split"
]

missing_columns = [
    col
    for col in required_columns
    if col not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# VERIFY IDENTITY-SPECIES UNIQUENESS
identity_species_counts = (
    metadata.groupby("identity")["species"]
    .nunique()
)

if (identity_species_counts > 1).any():
    raise ValueError(
        "One or more identities are associated with multiple species."
    )

# REMOVE RECORDS WITH MISSING IMAGES
missing_records = []

for index, row in metadata.iterrows():

    image_path = DATASET_DIR / row["path"]

    if not image_path.exists():
        missing_records.append(index)

metadata = metadata.drop(
    index=missing_records
).reset_index(drop=True)

print(f"Records available: {len(metadata):,}")
print(f"Missing image records removed: {len(missing_records):,}")

# VERIFY TRAIN / TEST SPLIT
if not {"train", "test"}.issubset(
    set(metadata["split"].unique())
):
    raise ValueError(
        "Metadata must contain train and test split values."
    )

# SPLIT DATASET
original_train = metadata[
    metadata["split"] == "train"
].copy()

original_test = metadata[
    metadata["split"] == "test"
].copy()

train_records, validation_records = train_test_split(
    original_train,
    test_size=VALIDATION_RATIO,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_records["experiment_split"] = "train"
validation_records["experiment_split"] = "validation"
original_test["experiment_split"] = "test"

final_split = pd.concat(
    [
        train_records,
        validation_records,
        original_test
    ],
    ignore_index=True
)

# VERIFY ORIGINAL TEST SET INTEGRITY
if set(original_test["path"]) != set(
    final_split[
        final_split["experiment_split"] == "test"
    ]["path"]
):
    raise ValueError(
        "Original test set was modified."
    )

# VERIFY IDENTITY OVERLAP
train_ids = set(
    train_records["identity"]
)

validation_ids = set(
    validation_records["identity"]
)

test_ids = set(
    original_test["identity"]
)

train_validation_overlap = (
    train_ids & validation_ids
)

train_test_overlap = (
    train_ids & test_ids
)

validation_test_overlap = (
    validation_ids & test_ids
)

print()
print("Identity overlap statistics")
print(
    f"Train/Validation: {len(train_validation_overlap):,}"
)
print(
    f"Train/Test: {len(train_test_overlap):,}"
)
print(
    f"Validation/Test: {len(validation_test_overlap):,}"
)

# SAVE FINAL SPLIT
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

final_split.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Train: {len(train_records):,}")
print(f"Validation: {len(validation_records):,}")
print(f"Test: {len(original_test):,}")
print(f"Saved: {OUTPUT_FILE}")
