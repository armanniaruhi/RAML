import torch
import yaml
import argparse
from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders, create_subset_loader
from src.ml.resNet50 import SiameseResNet
from src.ml.hyperparam_study import run_optuna_study
from pytorch_metric_learning.losses import ContrastiveLoss, MarginLoss, MultiSimilarityLoss, HistogramLoss

# Load config from YAML
with open("config/config.yml", "r") as f:
    config = yaml.safe_load(f)

# Extract sections
PRE = config["PREPROCESSING"]
TRAIN = config["TRAINING"]

# Set constants from preprocessing config
IMAGE_DIR = PRE["image_dir"]
LABEL_FILE = PRE["label_file"]
PARTITION_FILE = PRE["partition_file"]
BATCH_SIZE = PRE["batch_size"]
M_PER_SAMPLE = PRE["m_per_sample"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set constants from training config
LR = TRAIN["lr"]
WEIGHT_DECAY = TRAIN["weight_decay"]
NUM_EPOCHS = TRAIN["num_epochs"]
PATIENCE = TRAIN["patience"]
LOSS_TYPE = TRAIN["loss_type"]
HIDDEN_DIM = TRAIN["hidden_dimension"]
EMBEDDING_DIM = TRAIN["embedding_dimension"]

def main(mode):
    # Load datasets
    train_loader, val_loader, test_loader = get_partitioned_dataloaders(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        m_per_sample=M_PER_SAMPLE,
        partition_file=PARTITION_FILE,
        batch_size=BATCH_SIZE
    )

    if mode == "study":
        train_loader_study = create_subset_loader(train_loader, 50000)
        val_loader_study = create_subset_loader(train_loader, 8000)

        study = run_optuna_study(
            train_loader=train_loader_study,
            val_loader=val_loader_study,
            n_trials=50,
            study_name="siamese_contrastive_study",
            criterion="contrastive"
        )

        print("Best hyperparameters found:")
        print(study.best_params)


    elif mode == "train":
        model = SiameseResNet(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        if LOSS_TYPE == "contrastive":
            loss_fn = ContrastiveLoss(neg_margin=1.0, pos_margin=0)

        elif LOSS_TYPE == "histogram":
            loss_fn = HistogramLoss()

        elif LOSS_TYPE == "margin":
            loss_fn = MarginLoss()

        elif LOSS_TYPE == "multisimilarity":
            loss_fn = MultiSimilarityLoss()

        else:
            raise ValueError(f"Unsupported loss type: {LOSS_TYPE}")

        results = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=loss_fn,
            optimizer=optimizer,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            patience=PATIENCE,
            experiment_name='SiameseResNet',
            tuning_mode=False
        )

        print("Training complete.")

    else:
        raise ValueError("Invalid mode. Choose 'study' or 'train'.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", required=True, choices=["study", "train"], help="Run mode: 'study' or 'train'")
    # args = parser.parse_args()
    input = "train"
    # Pass the mode to main() function
    main(input)
