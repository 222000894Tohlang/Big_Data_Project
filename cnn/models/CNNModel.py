import os

import torch
import torch.nn as nn

import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


class SimpleCNN(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()


        self.features = nn.Sequential(

            # 128 -> 64
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 64 -> 32
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # 32 -> 16
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


class CNNTrainer:

    def __init__(
        self,
        num_classes,
        learning_rate=0.001,
        device=None
    ):

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = device


        self.model = SimpleCNN(
            num_classes
        ).to(self.device)


        self.criterion = (
            nn.CrossEntropyLoss()
        )


        self.optimizer = torch.optim.Adam(

            self.model.parameters(),

            lr=learning_rate
        )


        self.history = []


        print(
            "\nUsing device:",
            self.device
        )


    # --------------------------------------------------------
    # TRAIN ONE EPOCH
    # --------------------------------------------------------

    def train_epoch(
            self,
            loader,
            epoch,
            total_epochs
    ):

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs} [Train]",
            unit="batch"
        )

        for images, labels in progress:
            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # ----------------------------------------
            # Forward
            # ----------------------------------------

            self.optimizer.zero_grad()

            outputs = self.model(
                images
            )

            # ----------------------------------------
            # Loss
            # ----------------------------------------

            loss = self.criterion(
                outputs,
                labels
            )

            # ----------------------------------------
            # Backpropagation
            # ----------------------------------------

            loss.backward()

            self.optimizer.step()

            # ----------------------------------------
            # Statistics
            # ----------------------------------------

            batch_size = images.size(0)

            total_loss += (
                    loss.item() * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                    predictions == labels
            ).sum().item()

            total += batch_size

            # ----------------------------------------
            # Current statistics
            # ----------------------------------------

            current_loss = (
                    total_loss / total
            )

            current_accuracy = (
                    correct / total
            )

            progress.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_accuracy:.4f}"
            )

        epoch_loss = (
                total_loss / total
        )

        epoch_accuracy = (
                correct / total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(
            self,
            loader,
            epoch=None,
            total_epochs=None
    ):

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        if epoch is not None:

            description = (
                f"Epoch {epoch}/{total_epochs} "
                f"[Validation]"
            )

        else:

            description = "Validation"

        progress = tqdm(
            loader,
            desc=description,
            unit="batch"
        )

        with torch.no_grad():

            for images, labels in progress:
                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                outputs = self.model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

                batch_size = images.size(0)

                total_loss += (
                        loss.item() * batch_size
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                        predictions == labels
                ).sum().item()

                total += batch_size

                current_loss = (
                        total_loss / total
                )

                current_accuracy = (
                        correct / total
                )

                progress.set_postfix(
                    loss=f"{current_loss:.4f}",
                    acc=f"{current_accuracy:.4f}"
                )

        epoch_loss = (
                total_loss / total
        )

        epoch_accuracy = (
                correct / total
        )

        return (
            epoch_loss,
            epoch_accuracy
        )


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    def train(
            self,
            train_loader,
            validation_loader,
            epochs
    ):

        print("\nStarting training...\n")

        for epoch in range(1, epochs + 1):
            # ----------------------------------------
            # Training
            # ----------------------------------------

            train_loss, train_accuracy = (
                self.train_epoch(
                    train_loader,
                    epoch,
                    epochs
                )
            )

            # ----------------------------------------
            # Validation
            # ----------------------------------------

            validation_loss, validation_accuracy = (
                self.validate(
                    validation_loader,
                    epoch,
                    epochs
                )
            )

            # ----------------------------------------
            # Save history
            # ----------------------------------------

            record = {

                "epoch": epoch,

                "train_loss":
                    train_loss,

                "validation_loss":
                    validation_loss,

                "train_accuracy":
                    train_accuracy,

                "validation_accuracy":
                    validation_accuracy
            }

            self.history.append(
                record
            )

            # ----------------------------------------
            # Epoch summary
            # ----------------------------------------

            print()

            print(
                f"Epoch {epoch}/{epochs} Summary"
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Validation Loss: "
                f"{validation_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy * 100:.2f}%"
            )

            print(
                f"Validation Accuracy: "
                f"{validation_accuracy * 100:.2f}%"
            )

            print(
                "-" * 60
            )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    def evaluate(
            self,
            test_loader
    ):

        loss, accuracy = self.validate(
            test_loader
        )

        print(
            "\n================================"
        )

        print(
            "TEST RESULTS"
        )

        print(
            "================================"
        )

        print(
            f"Test Loss: {loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        return loss, accuracy


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    def save(
        self,
        path,
        species_to_id,
        image_size
    ):

        torch.save(

            {

                "model_state_dict":
                    self.model.state_dict(),

                "species_to_id":
                    species_to_id,

                "image_size":
                    image_size
            },

            path
        )


        print(
            "\nModel saved:"
        )

        print(path)


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    def save_history(
        self,
        path
    ):

        history_df = pd.DataFrame(
            self.history
        )

        history_df.to_csv(
            path,
            index=False
        )


    # --------------------------------------------------------
    # PLOT RESULTS
    # --------------------------------------------------------

    def plot_results(
        self,
        output_directory
    ):

        history_df = pd.DataFrame(
            self.history
        )


        # Loss

        plt.figure()

        plt.plot(
            history_df["epoch"],
            history_df["train_loss"],
            label="Training Loss"
        )

        plt.plot(
            history_df["epoch"],
            history_df["validation_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title(
            "CNN Training and Validation Loss"
        )

        plt.legend()

        plt.savefig(
            os.path.join(
                output_directory,
                "loss.png"
            )
        )

        plt.close()


        # Accuracy

        plt.figure()

        plt.plot(
            history_df["epoch"],
            history_df["train_accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            history_df["epoch"],
            history_df["validation_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.title(
            "CNN Training and Validation Accuracy"
        )

        plt.legend()

        plt.savefig(
            os.path.join(
                output_directory,
                "accuracy.png"
            )
        )

        plt.close()


    # --------------------------------------------------------
    # PREDICT IMAGE
    # --------------------------------------------------------

    def predict(
        self,
        image_path,
        transform,
        id_to_species
    ):

        self.model.eval()


        image = Image.open(
            image_path
        ).convert("RGB")


        image = transform(
            image
        )


        image = image.unsqueeze(0)


        image = image.to(
            self.device
        )


        with torch.no_grad():

            output = self.model(
                image
            )


            probabilities = torch.softmax(
                output,
                dim=1
            )


            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )


        predicted_species = (
            id_to_species[
                prediction.item()
            ]
        )


        return (
            predicted_species,
            confidence.item()
        )