
#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip

# Install CPU-only PyTorch
python -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI and SVM dependencies
python -m pip install -r api/requirements-deploy.txt

echo "Backend dependencies installed."
