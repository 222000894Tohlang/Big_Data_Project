import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import time
from config import FEATURES_PATH, MODEL_PATH

def main():
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val   = data["X_val"], data["y_val"]
    X_test, y_test  = data["X_test"], data["y_test"]
    species_list = list(data["species_list"])

    # Scale features
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)
    Xte = scaler.transform(X_test)

    # Hyperparameter search for C
    best_C, best_val_f1 = None, -1.0

    for C in [0.01, 0.1, 1]:
        clf = LinearSVC(
            C=C,class_weight="balanced",max_iter=5000,dual="auto",verbose=1
        )

        start = time.time()
        clf.fit(Xtr, y_train)
        elapsed = time.time() - start

        val_f1 = f1_score(
            y_val, clf.predict(Xva), average="macro"
        )
        val_acc = accuracy_score(y_val, clf.predict(Xva))

        print(
            f"C={C:>6} "
            f"val_acc={val_acc:.4f} "
            f"val_macroF1={val_f1:.4f} "
            f"iterations={clf.n_iter_} "
            f"time={elapsed:.2f}s",
            flush=True
        )

    # Train final model on train+val
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([y_train, y_val])
    final = LinearSVC(C=best_C, class_weight="balanced",
                      max_iter=5000, dual="auto").fit(Xtrva, ytrva)

    # Evaluate on test
    y_pred = final.predict(Xte)
    print("\n=== TEST (leakage-aware, unseen) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print()
    print(classification_report(y_test, y_pred, target_names=species_list, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    # Save the model, scaler, species list, and config
    joblib.dump({
        "scaler": scaler,
        "svm": final,
        "species_list": species_list,
        "config": {
            "C": best_C,
            "class_weight": "balanced",
            "img_size": list(__import__("config").IMG_SIZE),
            "hog_orientations": __import__("config").HOG_ORIENTATIONS,
            "hog_pixels_per_cell": list(__import__("config").HOG_PPC),
            "hog_cells_per_block": list(__import__("config").HOG_CPB),
            "hog_block_norm": __import__("config").HOG_BLOCK_NORM,
        },
    }, MODEL_PATH)
    print("\nModel saved:", MODEL_PATH)

if __name__ == "__main__":
    main()