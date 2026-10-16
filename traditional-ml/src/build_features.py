import time
import numpy as np
from tqdm import tqdm
from data_loader import load_splits
from hog_features import load_image_gray, extract_hog, hog_feature_dim
from config import FEATURES_PATH, LABEL_COL

def build_split_features(df, species_to_id):
    X = np.zeros((len(df), hog_feature_dim()), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int64)
    failed = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        try:
            gray = load_image_gray(getattr(row, "path"))
            X[i] = extract_hog(gray)
            y[i] = species_to_id[getattr(row, "species")]
        except Exception as e:
            failed.append((getattr(row, "path"), str(e)))
            y[i] = -1     # "skip"
    if failed:
        print(f"\n{len(failed)} images failed to load. First 5:")
        for p, e in failed[:5]:
            print(f"  {p}: {e}")
    keep = y >= 0
    return X[keep], y[keep]

def main():
    splits = load_splits()
    species_list = sorted(splits["train"]["species"].unique())
    species_to_id = {s: i for i, s in enumerate(species_list)}
    print("Classes:", species_to_id)

    t0 = time.time()
    X_train, y_train = build_split_features(splits["train"], species_to_id)
    X_val,   y_val   = build_split_features(splits["val"],   species_to_id)
    X_test,  y_test  = build_split_features(splits["test"],  species_to_id)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    np.savez_compressed(
        FEATURES_PATH,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        species_list=np.array(species_list),
    )
    print("Saved:", FEATURES_PATH)

if __name__ == "__main__":
    main()