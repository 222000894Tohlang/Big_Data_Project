# CNN

# 1. Model File
wildlife_cnn.pth
Location:
cnn\models\wildlife_cnn.pth

This contains the trained PyTorch model weights.
The API must recreate the same architecture before loading the weights.
The architecture is:
Input
128 × 128 × 3
       │
       ▼
Conv2D
3 → 32
       │
      ReLU
       │
   MaxPool
       │
       ▼
Conv2D
32 → 64
       │
      ReLU
       │
   MaxPool
       │
       ▼
Conv2D
64 → 128
       │
      ReLU
       │
   MaxPool
       │
       ▼
Flatten
       │
       ▼
Dense
128 × 16 × 16 → 256
       │
      ReLU
       │
    Dropout
       │
       ▼
Dense
256 → 22
       │
       ▼
22 Class Scores

# 2. Species Label Mapping
The model produces numeric class IDs.
The mapping is stored in:
cnn\results\species_labels.json

Current mapping:
{
    "0": "bird",
    "1": "cat",
    "2": "chicken",
    "3": "chimpanzee",
    "4": "cow",
    "5": "dog",
    "6": "dolphin",
    "7": "fish",
    "8": "giraffe",
    "9": "hyena",
    "10": "leopard",
    "11": "macaque",
    "12": "nyala",
    "13": "panda",
    "14": "polar bear",
    "15": "sea star",
    "16": "sea turtle",
    "17": "seal",
    "18": "tiger",
    "19": "whale",
    "20": "whaleshark",
    "21": "zebra"
}

The API should load this file when it starts.
For example:
with open("species_labels.json", "r") as f:    labels = json.load(f)id_to_species = {    int(k): v    for k, v in labels.items()}

