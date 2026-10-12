from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Maximum upload size: 10 MB
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# Supported image types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
}

# Frontend origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]