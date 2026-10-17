import os

from DataHandler import DataHandler
from CNNModel import CNNTrainer


# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_ROOT = r"D:\Studies\Dataset\archive (1)"

CSV_FILE = (
    r"D:\Studies\Dataset\archive (1)"
    r"\splits\final_original_split.csv"
)


MODEL_DIR = (
    r"D:\Studies\Dataset\archive (1)"
    r"\models"
)


RESULT_DIR = (
    r"D:\Studies\Dataset\archive (1)"
    r"\results"
)


IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 1

LEARNING_RATE = 0.001


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "WILDLIFE CNN BASELINE"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data = DataHandler(

        csv_file=CSV_FILE,

        image_root=IMAGE_ROOT,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE
    )


    (
        train_loader,
        validation_loader,
        test_loader
    ) = data.prepare()


    print(
        "\nNumber of species:",
        len(data.species)
    )


    # --------------------------------------------------------
    # SAVE LABELS
    # --------------------------------------------------------

    labels_file = os.path.join(
        RESULT_DIR,
        "species_labels.json"
    )


    data.save_labels(
        labels_file
    )


    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    trainer = CNNTrainer(

        num_classes=len(
            data.species
        ),

        learning_rate=LEARNING_RATE
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trainer.train(

        train_loader,

        validation_loader,

        EPOCHS
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    trainer.evaluate(
        test_loader
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_file = os.path.join(
        MODEL_DIR,
        "wildlife_cnn.pth"
    )


    trainer.save(

        model_file,

        data.species_to_id,

        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # SAVE TRAINING HISTORY
    # --------------------------------------------------------

    history_file = os.path.join(
        RESULT_DIR,
        "training_history.csv"
    )


    trainer.save_history(
        history_file
    )


    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    trainer.plot_results(
        RESULT_DIR
    )


    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    example_image = (
        data.df[
            data.df["experiment_split"]
            == "test"
        ].iloc[0]["path"]
    )


    _, test_transform = (
        data.create_transforms()
    )


    predicted_species, confidence = (
        trainer.predict(

            example_image,

            test_transform,

            data.id_to_species
        )
    )


    print(
        "\n=========================================="
    )

    print(
        "EXAMPLE PREDICTION"
    )

    print(
        "=========================================="
    )

    print(
        "Image:",
        example_image
    )

    print(
        "Prediction:",
        predicted_species
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )


    print(
        "\n=========================================="
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":

    main()