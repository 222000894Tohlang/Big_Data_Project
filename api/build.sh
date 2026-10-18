
#!/usr/bin/env bash
set -e

python3 -m pip install --upgrade pip

# Install CPU-only PyTorch
python3 -m pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu

# Install FastAPI and SVM dependencies
python3 -m pip install -r api/requirements-deploy.txt

echo "Backend dependencies installed."
