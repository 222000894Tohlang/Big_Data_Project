import os
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = Path("data/Dataset")
INPUT_FILE = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = Path("splits")

# Intermediate identity-level split
LEAKAGE_AWARE_OUTPUT = (
    OUTPUT_DIR / "leakage_aware_split.csv"
)

# Final cleaned split
FINAL_OUTPUT = (
    OUTPUT_DIR / "final_clean_split.csv"
)

# Summary files
SPECIES_SUMMARY_FILE = (
    OUTPUT_DIR / "final_clean_species_summary.csv"
)

IDENTITY_SUMMARY_FILE = (
    OUTPUT_DIR / "final_clean_identity_summary.csv"
)

# Duplicate audit
DUPLICATE_SUMMARY_FILE = (
    OUTPUT_DIR / "duplicate_summary.csv"
)

CROSS_SPLIT_DUPLICATES_FILE = (
    OUTPUT_DIR / "cross_split_duplicates.csv"
)

# Reproducibility
RANDOM_SEED = 42

# split proportions
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Hashing
HASH_CHUNK_SIZE = 1024 * 1024

def print_section(title):
    """Print a clear section heading."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# METADATA VALIDATION
def check_required_columns(metadata):
    """Check that the expected metadata columns exist."""

    required_columns = [
        "identity",
        "path",
        "date",
        "orientation",
        "species",
        "split",
        "dataset",
        "cluster_id"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print("Required metadata columns found.")


def check_identity_species_consistency(metadata):
    """
    Verify that each identity is associated with
    only one species.
    """

    print_section("CHECKING IDENTITY / SPECIES CONSISTENCY")

    identity_species = (
        metadata
        .groupby("identity")["species"]
        .nunique()
    )

    inconsistent = identity_species[
        identity_species > 1
    ]

    if len(inconsistent) > 0:

        print(
            f"ERROR: {len(inconsistent)} identities "
            "are associated with multiple species."
        )

        print(inconsistent)

        raise ValueError(
            "Identity/species consistency check failed."
        )

    print(
        "PASS: Every identity is associated with "
        "exactly one species."
    )


def check_image_paths(metadata):
    """
    Check whether image paths referenced by metadata exist.
    """

    print_section("CHECKING IMAGE PATHS")

    missing = []

    for path in metadata["path"]:

        image_path = DATASET_DIR / path

        if not image_path.exists():
            missing.append(path)

    if len(metadata) > 0:

        example_path = (
            DATASET_DIR / metadata.iloc[0]["path"]
        )

        print(
            "Example resolved image path:\n"
            f"   {example_path}"
        )

    if len(missing) == 0:

        print(
            f"PASS: All {len(metadata):,} image paths "
            "were found."
        )

    else:

        print(
            f"WARNING: {len(missing):,} image paths "
            "could not be found."
        )

        for path in missing[:10]:
            print("   ", path)

        if len(missing) > 10:

            print(
                f"   ... and {len(missing) - 10:,} more."
            )

    return missing

# IDENTITY-LEVEL SPLITTING
def create_identity_table(metadata):
    """
    Create one row per individual animal.
    """

    identity_table = (
        metadata[
            ["identity", "species"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return identity_table


def split_identities_by_species(identity_table):
    """
    Split identities separately within each species.

    All images belonging to an identity remain
    in the same partition.
    """

    print_section("CREATING IDENTITY-LEVEL SPLIT")

    train_ids = []
    validation_ids = []
    test_ids = []

    for species, group in identity_table.groupby("species"):

        species_ids = group["identity"].tolist()

        number_of_identities = len(species_ids)

        print(
            f"{species}: "
            f"{number_of_identities:,} identities"
        )

        if number_of_identities < 3:

            print(
                f"   WARNING: Only {number_of_identities} "
                "identities available."
            )

            train_ids.extend(species_ids)

            continue

        # 70% train, 30% temporary

        train, temporary = train_test_split(
            species_ids,
            test_size=(
                VALIDATION_RATIO + TEST_RATIO
            ),
            random_state=RANDOM_SEED
        )

        # Split temporary 50/50 into validation and test

        validation, test = train_test_split(
            temporary,
            test_size=0.5,
            random_state=RANDOM_SEED
        )

        train_ids.extend(train)
        validation_ids.extend(validation)
        test_ids.extend(test)

    return train_ids, validation_ids, test_ids


def create_split_mapping(
    train_ids,
    validation_ids,
    test_ids
):
    """Create identity -> split mapping."""

    identity_split = {}

    for identity in train_ids:
        identity_split[identity] = "train"

    for identity in validation_ids:
        identity_split[identity] = "validation"

    for identity in test_ids:
        identity_split[identity] = "test"

    return identity_split


def apply_split(metadata, identity_split):
    """Assign every record to its identity's split."""

    metadata = metadata.copy()

    metadata["leakage_aware_split"] = (
        metadata["identity"].map(identity_split)
    )

    return metadata


# SPLIT VALIDATION
def validate_all_images_assigned(metadata):
    """Ensure every record has exactly one split."""

    print_section("CHECKING SPLIT ASSIGNMENT")

    missing_split = metadata[
        metadata["leakage_aware_split"].isna()
    ]

    if len(missing_split) > 0:

        print(
            f"ERROR: {len(missing_split):,} records "
            "have no split."
        )

        raise ValueError(
            "Some images were not assigned to a split."
        )

    print(
        f"PASS: All {len(metadata):,} records "
        "have a split."
    )


def validate_identity_overlap(metadata):
    """
    Ensure no identity appears in more than one split.
    """

    print_section("CHECKING IDENTITY LEAKAGE")

    train_ids = set(
        metadata.loc[
            metadata["leakage_aware_split"] == "train",
            "identity"
        ]
    )

    validation_ids = set(
        metadata.loc[
            metadata["leakage_aware_split"] == "validation",
            "identity"
        ]
    )

    test_ids = set(
        metadata.loc[
            metadata["leakage_aware_split"] == "test",
            "identity"
        ]
    )

    train_validation = train_ids & validation_ids
    train_test = train_ids & test_ids
    validation_test = validation_ids & test_ids

    print(
        f"Train identities:       {len(train_ids):,}"
    )

    print(
        f"Validation identities:  {len(validation_ids):,}"
    )

    print(
        f"Test identities:        {len(test_ids):,}"
    )

    print()

    print(
        f"Train ∩ Validation:     "
        f"{len(train_validation)}"
    )

    print(
        f"Train ∩ Test:           "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test:      "
        f"{len(validation_test)}"
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):

        raise ValueError(
            "IDENTITY LEAKAGE DETECTED."
        )

    print()
    print(
        "PASS: No identity occurs in "
        "more than one split."
    )


# IMAGE HASHING
def calculate_md5(file_path):
    """
    Calculate the MD5 hash of an image file.
    """

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(HASH_CHUNK_SIZE)

            if not data:
                break

            md5.update(data)

    return md5.hexdigest()


def calculate_image_hashes(metadata):
    """
    Calculate an MD5 hash for every image.

    This allows exact duplicate images to be detected
    even when they have different paths or identities.
    """

    print_section("CALCULATING IMAGE HASHES")

    hashes = []
    failed = []

    total = len(metadata)

    for index, path in enumerate(
        metadata["path"],
        start=1
    ):

        image_path = DATASET_DIR / path

        try:

            image_hash = calculate_md5(
                image_path
            )

            hashes.append(image_hash)

        except Exception as error:

            print(
                f"\nERROR hashing image:\n"
                f"   {image_path}\n"
                f"   {error}"
            )

            hashes.append(None)
            failed.append(path)

        # Progress every 10,000 images

        if (
            index % 10000 == 0
            or index == total
        ):

            print(
                f"Hashed {index:,} / {total:,} images"
            )

    if failed:

        raise ValueError(
            f"{len(failed):,} images could not be hashed."
        )

    metadata = metadata.copy()

    metadata["image_hash"] = hashes

    print()
    print(
        f"PASS: Successfully hashed "
        f"{len(metadata):,} images."
    )

    print(
        f"Unique image hashes: "
        f"{metadata['image_hash'].nunique():,}"
    )

    return metadata

# DUPLICATE ANALYSIS
def analyze_duplicates(metadata):
    """
    Find exact duplicate images and determine which
    duplicates cross train/validation/test boundaries.
    """

    print_section("CHECKING EXACT DUPLICATES")

    hash_counts = (
        metadata["image_hash"]
        .value_counts()
    )

    duplicate_hashes = hash_counts[
        hash_counts > 1
    ].index

    duplicate_records = metadata[
        metadata["image_hash"].isin(
            duplicate_hashes
        )
    ].copy()

    duplicate_group_count = (
        len(duplicate_hashes)
    )

    duplicate_image_count = (
        len(duplicate_records)
    )

    print(
        f"Duplicate hash groups: "
        f"{duplicate_group_count:,}"
    )

    print(
        f"Images in duplicate groups: "
        f"{duplicate_image_count:,}"
    )

    # Find hashes occurring in multiple splits
    hash_split_counts = (
        duplicate_records
        .groupby("image_hash")
        ["leakage_aware_split"]
        .nunique()
    )

    cross_split_hashes = hash_split_counts[
        hash_split_counts > 1
    ].index

    cross_split_duplicates = duplicate_records[
        duplicate_records["image_hash"].isin(
            cross_split_hashes
        )
    ].copy()

    print(
        f"Cross-split duplicate groups: "
        f"{len(cross_split_hashes):,}"
    )

    print(
        f"Records involved in cross-split "
        f"duplicates: "
        f"{len(cross_split_duplicates):,}"
    )

    # Save cross-split duplicate audit
    audit_columns = [
        "image_hash",
        "path",
        "identity",
        "species",
        "leakage_aware_split"
    ]

    cross_split_duplicates[
        audit_columns
    ].to_csv(
        CROSS_SPLIT_DUPLICATES_FILE,
        index=False
    )

    print(
        f"\nSaved cross-split duplicate audit to:\n"
        f"{CROSS_SPLIT_DUPLICATES_FILE}"
    )

    # Save duplicate summary
    duplicate_summary = pd.DataFrame({
        "total_images": [len(metadata)],
        "unique_hashes": [
            metadata["image_hash"].nunique()
        ],
        "duplicate_groups": [
            duplicate_group_count
        ],
        "images_in_duplicate_groups": [
            duplicate_image_count
        ],
        "cross_split_duplicate_groups": [
            len(cross_split_hashes)
        ],
        "cross_split_duplicate_records": [
            len(cross_split_duplicates)
        ]
    })

    duplicate_summary.to_csv(
        DUPLICATE_SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved duplicate summary to:\n"
        f"{DUPLICATE_SUMMARY_FILE}"
    )

    return cross_split_duplicates


# CROSS-SPLIT DUPLICATE CLEANING
def remove_cross_split_duplicates(
    metadata,
    cross_split_duplicates
):

    print_section(
        "REMOVING CROSS-SPLIT EXACT DUPLICATES"
    )

    split_priority = {
        "train": 0,
        "validation": 1,
        "test": 2
    }

    paths_to_remove = set()

    for image_hash, group in (
        cross_split_duplicates
        .groupby("image_hash")
    ):

        group = group.copy()

        group["split_priority"] = (
            group["leakage_aware_split"]
            .map(split_priority)
        )

        highest_priority = (
            group["split_priority"].min()
        )

        records_to_remove = group[
            group["split_priority"]
            > highest_priority
        ]

        for _, row in (
            records_to_remove.iterrows()
        ):

            paths_to_remove.add(
                row["path"]
            )

    print(
        f"Records selected for removal: "
        f"{len(paths_to_remove):,}"
    )

    # Remove only the selected paths

    metadata = metadata.copy()

    metadata["remove_duplicate"] = (
        metadata["path"]
        .isin(paths_to_remove)
    )

    removed = metadata[
        metadata["remove_duplicate"]
    ].copy()

    final_metadata = metadata[
        ~metadata["remove_duplicate"]
    ].copy()

    final_metadata.drop(
        columns=["remove_duplicate"],
        inplace=True
    )

    print(
        f"Records removed: "
        f"{len(removed):,}"
    )

    print(
        f"Final records: "
        f"{len(final_metadata):,}"
    )

    if len(removed) > 0:

        print("\nRemoved records by split:")

        print(
            removed[
                "leakage_aware_split"
            ]
            .value_counts()
            .sort_index()
        )

    return final_metadata, removed


# FINAL DUPLICATE VALIDATION
def validate_no_cross_split_duplicates(
    final_metadata
):
    """
    Verify that no exact image hash appears in more
    than one final partition.
    """

    print_section(
        "FINAL EXACT DUPLICATE LEAKAGE CHECK"
    )

    hash_split_counts = (
        final_metadata
        .groupby("image_hash")
        ["leakage_aware_split"]
        .nunique()
    )

    cross_split_hashes = hash_split_counts[
        hash_split_counts > 1
    ]

    print(
        f"Hashes across multiple splits: "
        f"{len(cross_split_hashes)}"
    )

    if len(cross_split_hashes) > 0:

        raise ValueError(
            "EXACT DUPLICATE LEAKAGE DETECTED "
            "IN FINAL DATASET."
        )

    print(
        "PASS: No exact duplicate image occurs "
        "in more than one split."
    )

# FINAL STATISTICS
def print_final_statistics(metadata):
    """
    Print final image and identity statistics.
    """

    print_section("FINAL SPLIT STATISTICS")

    split_counts = (
        metadata[
            "leakage_aware_split"
        ]
        .value_counts()
    )

    total = len(metadata)

    for split in [
        "train",
        "validation",
        "test"
    ]:

        count = split_counts.get(
            split,
            0
        )

        percentage = (
            count / total * 100
        )

        print(
            f"{split.capitalize():12}"
            f"{count:>12,} images"
            f" ({percentage:.2f}%)"
        )

    print()

    print(
        f"Total images: "
        f"{total:,}"
    )

    identity_counts = (
        metadata[
            [
                "identity",
                "leakage_aware_split"
            ]
        ]
        .drop_duplicates()
        ["leakage_aware_split"]
        .value_counts()
    )

    print()
    print("Identity distribution:")

    for split in [
        "train",
        "validation",
        "test"
    ]:

        count = identity_counts.get(
            split,
            0
        )

        print(
            f"{split.capitalize():12}"
            f"{count:>12,}"
        )


def print_species_distribution(metadata):
    """
    Print image counts by species and split.
    """

    print_section(
        "FINAL SPECIES DISTRIBUTION"
    )

    table = pd.crosstab(
        metadata["species"],
        metadata["leakage_aware_split"]
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        if split not in table.columns:
            table[split] = 0

    table = table[
        [
            "train",
            "validation",
            "test"
        ]
    ]

    table["total"] = table.sum(axis=1)

    print(
        table.to_string()
    )


# SUMMARY FILES
def create_species_summary(metadata):
    """
    Create image and identity counts for every
    species and split.
    """

    image_counts = (
        metadata
        .groupby(
            [
                "species",
                "leakage_aware_split"
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    identity_counts = (
        metadata[
            [
                "species",
                "identity",
                "leakage_aware_split"
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "species",
                "leakage_aware_split"
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    summary = pd.DataFrame(
        index=image_counts.index
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        summary[
            f"{split}_images"
        ] = image_counts.get(
            split,
            0
        )

        summary[
            f"{split}_identities"
        ] = identity_counts.get(
            split,
            0
        )

    summary = summary.reset_index()

    return summary


def create_identity_summary(metadata):
    """
    Create one row per identity.
    """

    return (
        metadata
        .groupby(
            [
                "identity",
                "species",
                "leakage_aware_split"
            ]
        )
        .size()
        .reset_index(
            name="image_count"
        )
    )


def main():

    print_section(
        "WILDLIFEREID-10K DATA PREPROCESSING"
    )

    print(
        f"Input file: {INPUT_FILE}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        "Target split: "
        f"{TRAIN_RATIO:.0%} train / "
        f"{VALIDATION_RATIO:.0%} validation / "
        f"{TEST_RATIO:.0%} test"
    )

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load metadata
    print_section("LOADING METADATA")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{INPUT_FILE}"
        )

    metadata = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"Loaded {len(metadata):,} metadata records."
    )

    print(
        f"Columns: {list(metadata.columns)}"
    )

    # Required metadata checks
    check_required_columns(
        metadata
    )

    check_identity_species_consistency(
        metadata
    )

    # Check image paths
    missing_paths = check_image_paths(
        metadata
    )

    if missing_paths:

        metadata = metadata[
            ~metadata["path"].isin(
                missing_paths
            )
        ].copy()

        print(
            f"\nExcluded {len(missing_paths):,} "
            "missing image records."
        )

    print(
        f"Records available for splitting: "
        f"{len(metadata):,}"
    )

    # Create identity table
    print_section(
        "CREATING IDENTITY-LEVEL DATA"
    )

    identity_table = create_identity_table(
        metadata
    )

    print(
        f"Unique identities: "
        f"{len(identity_table):,}"
    )

    print(
        f"Species: "
        f"{identity_table['species'].nunique()}"
    )

    # Identity-level split
    (
        train_ids,
        validation_ids,
        test_ids
    ) = split_identities_by_species(
        identity_table
    )

    print()
    print(
        f"Train identities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation identities: "
        f"{len(validation_ids):,}"
    )

    print(
        f"Test identities: "
        f"{len(test_ids):,}"
    )

    # Ensure every identity is assigned exactly once
    all_assigned_ids = (
        train_ids
        + validation_ids
        + test_ids
    )

    if len(all_assigned_ids) != len(
        identity_table
    ):

        raise ValueError(
            "Not every identity was assigned."
        )

    if len(set(all_assigned_ids)) != len(
        all_assigned_ids
    ):

        raise ValueError(
            "An identity was assigned more than once."
        )

    print(
        "PASS: Every identity assigned exactly once."
    )

    # Apply identity split
    identity_split = create_split_mapping(
        train_ids,
        validation_ids,
        test_ids
    )

    metadata = apply_split(
        metadata,
        identity_split
    )

    # Validate identity split
    validate_all_images_assigned(
        metadata
    )

    validate_identity_overlap(
        metadata
    )

    # Save identity-level split BEFORE deduplication
    metadata.to_csv(
        LEAKAGE_AWARE_OUTPUT,
        index=False
    )

    print(
        f"\nSaved identity-level split to:\n"
        f"{LEAKAGE_AWARE_OUTPUT}"
    )

    # Hash every image
    metadata = calculate_image_hashes(
        metadata
    )

    # Find cross-split duplicates
    cross_split_duplicates = analyze_duplicates(
        metadata
    )

    # Remove cross-split duplicates
    final_metadata, removed = (
        remove_cross_split_duplicates(
            metadata,
            cross_split_duplicates
        )
    )

    # Validate final identity leakage again
    validate_identity_overlap(
        final_metadata
    )

    # Validate final exact duplicate leakage
    validate_no_cross_split_duplicates(
        final_metadata
    )

    # Final statistics
    print_final_statistics(
        final_metadata
    )

    print_species_distribution(
        final_metadata
    )

    species_summary = create_species_summary(
        final_metadata
    )

    identity_summary = create_identity_summary(
        final_metadata
    )

    # Save final cleaned split
    final_metadata.to_csv(
        FINAL_OUTPUT,
        index=False
    )

    print(
        f"\nSaved FINAL clean split to:\n"
        f"{FINAL_OUTPUT}"
    )

    # Save final summaries
    species_summary.to_csv(
        SPECIES_SUMMARY_FILE,
        index=False
    )

    identity_summary.to_csv(
        IDENTITY_SUMMARY_FILE,
        index=False
    )

    print(
        f"Saved species summary to:\n"
        f"{SPECIES_SUMMARY_FILE}"
    )

    print(
        f"Saved identity summary to:\n"
        f"{IDENTITY_SUMMARY_FILE}"
    )

    # Final summary
    print_section(
        "FINAL PREPROCESSING SUMMARY"
    )

    print(
        f"Original metadata records:       "
        f"{len(pd.read_csv(INPUT_FILE, low_memory=False)):,}"
    )

    print(
        f"Missing image records excluded:  "
        f"{len(missing_paths):,}"
    )

    print(
        f"Records before duplicate cleaning:"
        f" {len(metadata):,}"
    )

    print(
        f"Cross-split duplicate groups:     "
        f"{cross_split_duplicates['image_hash'].nunique():,}"
    )

    print(
        f"Records removed:                  "
        f"{len(removed):,}"
    )

    print(
        f"Final records:                    "
        f"{len(final_metadata):,}"
    )

    print(
        "Identity leakage:                 0"
    )

    print(
        "Exact duplicate leakage:          0"
    )

    print()
    print(
        "FINAL DATASET IS READY FOR MODELING."
    )

    print_section(
        "PREPROCESSING COMPLETE"
    )

if __name__ == "__main__":
    main()