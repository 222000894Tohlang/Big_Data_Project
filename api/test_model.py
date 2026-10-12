from model_loader import (
    load_model,
    get_class_names,
    get_image_size,
    get_device,
)

model = load_model()

print("\nModel loaded:", type(model).__name__)

print("Device:", get_device())

print("Image size:", get_image_size())

print("\nSpecies classes:")

for index, species in get_class_names().items():
    print(f"{index}: {species}")