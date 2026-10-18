
import importlib.util

packages = [
    "numpy",
    "PIL",
    "skimage",
    "sklearn",
    "joblib",
    "cuml",
    "cupy",
]

print("Traditional ML environment check\n")

for package in packages:
    available = importlib.util.find_spec(package) is not None
    print(f"{package}: {'Available' if available else 'Not installed'}")
