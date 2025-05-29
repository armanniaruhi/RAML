import torch
import argparse
from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders, create_subset_loader
from src.ml.resNet50 import SiameseResNet
from src.ml.hyperparam_study import run_optuna_study
from pytorch_metric_learning.losses import ContrastiveLoss, MarginLoss, MultiSimilarityLoss, HistogramLoss

# Constants
IMAGE_DIR = "data/celeba/img_align_celeba"
LABEL_FILE = "data/celeba/identity_CelebA.txt"
PARTITION_FILE = "data/celeba/list_eval_partition.csv"
IMG_SIZE = 224
BATCH_SIZE = 16
M_PER_SAMPLE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        model = SiameseResNet().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        contrastive_loss = ContrastiveLoss(neg_margin=1.0, pos_margin=0)

        results = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=contrastive_loss,
            optimizer=optimizer,
            num_epochs=10,
            device=DEVICE,
            patience=5,
            experiment_name='SiameseResNet',
            tuning_mode=False
        )

        print("Training complete.")

    else:
        raise ValueError("Invalid mode. Choose 'study' or 'train'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["study", "train"], help="Run mode: 'study' or 'train'")
    args = parser.parse_args()

    # Pass the mode to main() function
    main(args.mode)
