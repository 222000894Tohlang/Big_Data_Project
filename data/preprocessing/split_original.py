import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = Path("data/Dataset")
INPUT_FILE = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = Path("splits")

FINAL_OUTPUT = OUTPUT_DIR / "final_original_split.csv"
SPECIES_SUMMARY_FILE = OUTPUT_DIR / "final_original_species_summary.csv"
IDENTITY_SUMMARY_FILE = OUTPUT_DIR / "final_original_identity_summary.csv"

RANDOM_SEED = 42

ORIGINAL_TRAIN_RATIO = 0.85
ORIGINAL_VALIDATION_RATIO = 0.15


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# LOAD METADATA
print_section("WILDLIFEREID-10K ORIGINAL DATA SPLIT")

print(f"Input file: {INPUT_FILE}")
print(f"Random seed: {RANDOM_SEED}")
print("Original dataset split will be preserved.")
print("Original train will be split into 85% train / 15% validation.")


print_section("LOADING METADATA")

metadata = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Loaded {len(metadata):,} metadata records.")
print(f"Columns: {list(metadata.columns)}")

# CHECK REQUIRED COLUMNS
required_columns = [
    "identity",
    "path",
    "species",
    "split"
]

missing_columns = [
    column for column in required_columns
    if column not in metadata.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required metadata columns found.")

# CHECK IDENTITY / SPECIES CONSISTENCY
print_section("CHECKING IDENTITY / SPECIES CONSISTENCY")

identity_species_counts = (
    metadata.groupby("identity")["species"]
    .nunique()
)

inconsistent_identities = identity_species_counts[
    identity_species_counts > 1
]

if len(inconsistent_identities) > 0:
    print(
        f"WARNING: {len(inconsistent_identities)} identities "
        "are associated with multiple species."
    )
else:
    print(
        "PASS: Every identity is associated with exactly one species."
    )

# CHECK IMAGE PATHS
print_section("CHECKING IMAGE PATHS")

def resolve_image_path(relative_path):
    return DATASET_DIR / relative_path


example_path = resolve_image_path(
    metadata.iloc[0]["path"]
)

print("Example resolved image path:")
print(f"   {example_path}")


missing_records = []

for index, row in metadata.iterrows():

    image_path = resolve_image_path(row["path"])

    if not image_path.exists():
        missing_records.append(index)


if missing_records:

    print(
        f"WARNING: {len(missing_records)} image paths "
        "could not be found."
    )

    for index in missing_records:
        print(f"    {metadata.loc[index, 'path']}")

else:

    print("PASS: All image paths were found.")


# Remove records whose image files do not exist.

metadata = metadata.drop(
    index=missing_records
).reset_index(drop=True)

print()
print(
    f"Excluded {len(missing_records)} missing image records."
)

print(
    f"Records available for original split: "
    f"{len(metadata):,}"
)

# CHECK ORIGINAL SPLIT VALUES
print_section("CHECKING ORIGINAL DATASET SPLIT")

print("Original split distribution:")

print(
    metadata["split"]
    .value_counts()
    .sort_index()
)


# Make sure the expected original split values exist.

if not set(["train", "test"]).issubset(
    set(metadata["split"].unique())
):
    raise ValueError(
        "The metadata does not contain the expected "
        "'train' and 'test' split values."
    )

# SEPARATE ORIGINAL TRAIN AND TEST
original_train = metadata[
    metadata["split"] == "train"
].copy()

original_test = metadata[
    metadata["split"] == "test"
].copy()


print()
print(f"Original train records: {len(original_train):,}")
print(f"Original test records:  {len(original_test):,}")

# CREATE VALIDATION SET
print_section("CREATING ORIGINAL TRAIN / VALIDATION SPLIT")

print(
    "The original test set will remain completely untouched."
)

print(
    "Only the original training records will be split."
)

print(
    f"Training ratio:    {ORIGINAL_TRAIN_RATIO:.0%}"
)

print(
    f"Validation ratio:  {ORIGINAL_VALIDATION_RATIO:.0%}"
)

original_train_records, original_validation_records = (
    train_test_split(
        original_train,
        test_size=ORIGINAL_VALIDATION_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )
)


print()
print(
    f"Original training records: "
    f"{len(original_train_records):,}"
)

print(
    f"Original validation records: "
    f"{len(original_validation_records):,}"
)

print(
    f"Original test records: "
    f"{len(original_test):,}"
)

# ASSIGN EXPERIMENT SPLIT LABELS
original_train_records = original_train_records.copy()
original_validation_records = original_validation_records.copy()
original_test = original_test.copy()

original_train_records[
    "experiment_split"
] = "train"

original_validation_records[
    "experiment_split"
] = "validation"

original_test[
    "experiment_split"
] = "test"

# COMBINE FINAL ORIGINAL SPLIT
final_split = pd.concat(
    [
        original_train_records,
        original_validation_records,
        original_test
    ],
    ignore_index=True
)


# Sort by split and then path to make the output easier to inspect.

split_order = {
    "train": 0,
    "validation": 1,
    "test": 2
}

final_split["_split_order"] = (
    final_split["experiment_split"]
    .map(split_order)
)

final_split = (
    final_split
    .sort_values(
        by=["_split_order", "path"]
    )
    .drop(columns=["_split_order"])
    .reset_index(drop=True)
)

# CHECK ALL RECORDS HAVE A SPLIT
print_section("CHECKING SPLIT ASSIGNMENT")

missing_split = final_split[
    final_split["experiment_split"].isna()
]

if len(missing_split) > 0:
    raise ValueError(
        f"{len(missing_split)} records do not have a split."
    )

print(
    f"PASS: All {len(final_split):,} records have a split."
)

# CHECK THAT ORIGINAL TEST WAS PRESERVED
print_section("CHECKING ORIGINAL TEST SET")

original_test_paths = set(
    original_test["path"]
)

final_test_paths = set(
    final_split[
        final_split["experiment_split"] == "test"
    ]["path"]
)

if original_test_paths != final_test_paths:

    raise ValueError(
        "ERROR: The original test set was changed."
    )

print(
    "PASS: Original test set was preserved unchanged."
)

print(
    f"Original test images: "
    f"{len(original_test_paths):,}"
)

print(
    f"Final test images:    "
    f"{len(final_test_paths):,}"
)

# CHECK IDENTITY OVERLAP
print_section("CHECKING IDENTITY OVERLAP")

train_identities = set(
    final_split[
        final_split["experiment_split"] == "train"
    ]["identity"]
)

validation_identities = set(
    final_split[
        final_split["experiment_split"] == "validation"
    ]["identity"]
)

test_identities = set(
    final_split[
        final_split["experiment_split"] == "test"
    ]["identity"]
)


print(
    f"Train identities:       {len(train_identities):,}"
)

print(
    f"Validation identities:  {len(validation_identities):,}"
)

print(
    f"Test identities:        {len(test_identities):,}"
)


train_validation_overlap = (
    train_identities & validation_identities
)

train_test_overlap = (
    train_identities & test_identities
)

validation_test_overlap = (
    validation_identities & test_identities
)


print()
print(
    f"Train ∩ Validation:     "
    f"{len(train_validation_overlap):,}"
)

print(
    f"Train ∩ Test:           "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation ∩ Test:      "
    f"{len(validation_test_overlap):,}"
)

total_identity_overlaps = (
    len(train_validation_overlap)
    + len(train_test_overlap)
    + len(validation_test_overlap)
)

print()

if total_identity_overlaps > 0:

    print(
        "EXPECTED BASELINE RESULT:"
    )

    print(
        "Identity overlap exists between partitions."
    )

    print(
        "This is intentionally retained because this split "
        "represents the original dataset methodology."
    )

else:

    print(
        "NOTE: No identity overlap was detected in this "
        "random split."
    )

# FINAL SPLIT STATISTICS
print_section("FINAL ORIGINAL SPLIT STATISTICS")

split_counts = (
    final_split["experiment_split"]
    .value_counts()
    .reindex(
        ["train", "validation", "test"]
    )
)

total_records = len(final_split)

for split_name, count in split_counts.items():

    percentage = (
        count / total_records
    ) * 100

    print(
        f"{split_name.capitalize():<15}"
        f"{count:>10,} images "
        f"({percentage:.2f}%)"
    )

print()
print(
    f"Total images:    {total_records:,}"
)

# SPECIES DISTRIBUTION
print_section("FINAL SPECIES DISTRIBUTION")

species_summary = (
    final_split
    .groupby(
        ["species", "experiment_split"]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reindex(
        columns=["train", "validation", "test"],
        fill_value=0
    )
)

species_summary["total"] = (
    species_summary[
        ["train", "validation", "test"]
    ].sum(axis=1)
)

species_summary = species_summary.sort_values(
    "total",
    ascending=False
)

print(species_summary)

# IDENTITY DISTRIBUTION
identity_summary = (
    final_split
    .groupby(
        ["identity", "species", "experiment_split"]
    )
    .size()
    .reset_index(
        name="image_count"
    )
)

# SAVE OUTPUTS
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


final_split.to_csv(
    FINAL_OUTPUT,
    index=False
)

species_summary.to_csv(
    SPECIES_SUMMARY_FILE
)

identity_summary.to_csv(
    IDENTITY_SUMMARY_FILE,
    index=False
)


print()
print(
    "Saved FINAL original split to:"
)

print(
    FINAL_OUTPUT
)

print()
print(
    "Saved species summary to:"
)

print(
    SPECIES_SUMMARY_FILE
)

print()
print(
    "Saved identity summary to:"
)

print(
    IDENTITY_SUMMARY_FILE
)

print_section("FINAL ORIGINAL SPLIT SUMMARY")

print(
    f"Original metadata records:       "
    f"{len(pd.read_csv(INPUT_FILE, low_memory=False)):,}"
)

print(
    f"Missing image records excluded:  "
    f"{len(missing_records):,}"
)

print(
    f"Records available:               "
    f"{len(metadata):,}"
)

print(
    f"Final records:                   "
    f"{len(final_split):,}"
)

print(
    f"Train records:                   "
    f"{len(original_train_records):,}"
)

print(
    f"Validation records:              "
    f"{len(original_validation_records):,}"
)

print(
    f"Test records:                    "
    f"{len(original_test):,}"
)

print()
print(
    f"Train/Validation identity overlap: "
    f"{len(train_validation_overlap):,}"
)

print(
    f"Train/Test identity overlap:       "
    f"{len(train_test_overlap):,}"
)

print(
    f"Validation/Test identity overlap:   "
    f"{len(validation_test_overlap):,}"
)

print()
print(
    "ORIGINAL BASELINE SPLIT COMPLETE."
)

print(
    "This split intentionally preserves the original "
    "dataset's leakage characteristics."
)

print(
    "=" * 70
)