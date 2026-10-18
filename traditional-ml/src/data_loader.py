import pandas as pd
from config import CSV_PATH, SPLIT_COL, PATH_COL, LABEL_COL

_SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}

# Load the splits from the CSV file and return a dictionary of DataFrames for each split
def load_splits(csv_path=CSV_PATH, split_col=SPLIT_COL,path_col=PATH_COL, label_col=LABEL_COL):
    df = pd.read_csv(csv_path, low_memory=False)

    # Normalize split names so both files work with the same loader
    df["_split"] = df[split_col].map(_SPLIT_MAP)
    if df["_split"].isna().any():
        bad = df.loc[df["_split"].isna(), split_col].unique().tolist()
        raise ValueError(f"Unexpected split values in '{split_col}': {bad}")

    out = {}
    for name in ("train", "val", "test"):
        out[name] = df[df["_split"] == name].reset_index(drop=True)
    return out

# Return a sorted list of unique species
def species_mapping(df):
    return sorted(df["species"].unique())

if __name__ == "__main__":
    splits = load_splits()
    for k, v in splits.items():
        print(f"{k}: {len(v)} images, {v['species'].nunique()} species")