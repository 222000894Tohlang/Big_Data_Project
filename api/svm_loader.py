
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SVM_MODEL_PATH = (
    PROJECT_ROOT
    / "traditional-ml"
    / "models"
    / "svm_model_sklearn.joblib"
)


class SVMUnavailableError(Exception):
    pass


def svm_model_exists() -> bool:
    return SVM_MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def load_svm_model():
    """
    Load a trusted, CPU-compatible SVM artifact.

    """
    if not svm_model_exists():
        raise SVMUnavailableError(
            "CPU-compatible SVM model is not available."
        )

    import joblib

    return joblib.load(SVM_MODEL_PATH)
