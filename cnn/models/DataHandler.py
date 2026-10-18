import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# A custom PyTorch Dataset for loading wildlife images and their corresponding species labels.
class WildlifeDataset(Dataset):
    def __init__(
            self,
            dataframe,
            species_to_id,
            image_root,
            transform=None
    ):
        self.df = dataframe.reset_index(drop=True)
        self.species_to_id = species_to_id
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # Get an image and its corresponding label by index. The image is loaded from disk, transformed, and returned along with its label.
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row["path"]

        if os.path.isabs(image_path):
            full_path = image_path
        else:
            full_path = os.path.join(
                self.image_root,
                image_path
            )

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Image not found:\n"
                f"CSV path: {image_path}\n"
                f"Resolved path: {full_path}"
            )

        image = Image.open(
            full_path
        ).convert("RGB")

        species = row["species"]

        label = self.species_to_id[
            species
        ]

        if self.transform:
            image = self.transform(
                image
            )

        return image, label


# A class to handle loading and preparing wildlife image data for training, validation, and testing. It loads metadata from a CSV file, creates data loaders, and applies transformations to the images.
class DataHandler:
    def __init__(
            self,
            csv_file,
            image_root,
            image_size=128,
            batch_size=32
    ):
        self.csv_file = csv_file
        self.image_root = image_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.df = None
        self.species = None
        self.species_to_id = None
        self.id_to_species = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None

    # LOAD METADATA
    def load_metadata(self):
        print("\nLoading metadata...")

        self.df = pd.read_csv(
            self.csv_file
        )

        self.species = sorted(
            self.df["species"].unique()
        )

        self.species_to_id = {
            species: i
            for i, species
            in enumerate(self.species)
        }

        self.id_to_species = {
            i: species
            for species, i
            in self.species_to_id.items()
        }

        print(
            "Images:",
            len(self.df)
        )

        print(
            "Species:",
            len(self.species)
        )

    # CREATE TRANSFORMS
    def create_transforms(self):
        train_transform = transforms.Compose([

            transforms.Resize(
                (
                    self.image_size,
                    self.image_size
                )
            ),

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize(
                (
                    self.image_size,
                    self.image_size
                )
            ),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        return (
            train_transform,
            test_transform
        )

    # CREATE DATA LOADERS
    def create_loaders(self):

        train_transform, test_transform = (
            self.create_transforms()
        )

        train_df = self.df[
            self.df["experiment_split"]
            == "train"
        ]

        validation_df = self.df[
            self.df["experiment_split"]
            == "validation"
        ]

        test_df = self.df[
            self.df["experiment_split"]
            == "test"
        ]

        print("\nDataset split:")

        print(
            "Train:",
            len(train_df)
        )

        print(
            "Validation:",
            len(validation_df)
        )

        print(
            "Test:",
            len(test_df)
        )

        train_dataset = WildlifeDataset(
            train_df,
            self.species_to_id,
            self.image_root,
            train_transform
        )

        validation_dataset = WildlifeDataset(
            validation_df,
            self.species_to_id,
            self.image_root,
            test_transform
        )

        test_dataset = WildlifeDataset(
            test_df,
            self.species_to_id,
            self.image_root,
            test_transform
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    # SAVE SPECIES LABELS
    def save_labels(self, file_path):
        with open(
            file_path,
            "w"
        ) as file:
            json.dump(
                self.id_to_species,
                file,
                indent=4
            )

    # PREPARE DATA
    def prepare(self):
        self.load_metadata()
        self.create_loaders()

        return (
            self.train_loader,
            self.validation_loader,
            self.test_loader
        )