from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# CONSTANTS
DATASET_DIR = Path("data/Dataset")
INPUT_FILE = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = Path("splits")
OUTPUT_FILE = OUTPUT_DIR / "leakage_aware_split.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# LOAD METADATA
metadata = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

required_columns = [
    "identity",
    "path",
    "species"
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
        "One or more identities belong to multiple species."
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
identity_table = (
    metadata[
        ["identity", "species"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

# SPLIT DATASET
train_ids = []
validation_ids = []
test_ids = []

for species, group in identity_table.groupby("species"):

    identities = group["identity"].tolist()

    if len(identities) < 3:

        train_ids.extend(identities)
        continue

    train, temp = train_test_split(
        identities,
        test_size=(
            VALIDATION_RATIO + TEST_RATIO
        ),
        random_state=RANDOM_SEED
    )

    validation, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    train_ids.extend(train)
    validation_ids.extend(validation)
    test_ids.extend(test)

# MAP IDENTITIES TO SPLITS
identity_split = {}

for identity in train_ids:
    identity_split[identity] = "train"

for identity in validation_ids:
    identity_split[identity] = "validation"

for identity in test_ids:
    identity_split[identity] = "test"

# ASSIGN SPLITS TO METADATA
metadata["experiment_split"] = (
    metadata["identity"]
    .map(identity_split)
)

if metadata["experiment_split"].isna().any():
    raise ValueError(
        "Some identities were not assigned a split."
    )

# CHECK FOR IDENTITY LEAKAGE
train_identity_set = set(train_ids)
validation_identity_set = set(validation_ids)
test_identity_set = set(test_ids)

if (
    train_identity_set & validation_identity_set
    or train_identity_set & test_identity_set
    or validation_identity_set & test_identity_set
):
    raise ValueError(
        "Identity leakage detected."
    )

print()
print("Identity overlap statistics")
print(
    f"Train/Validation: "
    f"{len(train_identity_set & validation_identity_set)}"
)
print(
    f"Train/Test: "
    f"{len(train_identity_set & test_identity_set)}"
)
print(
    f"Validation/Test: "
    f"{len(validation_identity_set & test_identity_set)}"
)

# SAVE FINAL SPLIT
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

metadata.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Train identities: {len(train_ids):,}")
print(f"Validation identities: {len(validation_ids):,}")
print(f"Test identities: {len(test_ids):,}")
print(f"Saved: {OUTPUT_FILE}")