import optuna
from optuna.samplers import TPESampler
from src.ml.resNet50 import SiameseResNet
import torch

from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.losses import MarginLoss
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.losses import HistogramLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def objective(trial, train_loader, val_loader, criterion):
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    hidden_dims = trial.suggest_categorical("hidden_dims", [
    (1024, 512), 
    (1024, 512, 256),  
    (512, 256)
    ])

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])

    # Initialize model
    model = SiameseResNet(embedding_dim=embedding_dim, hidden_dim=hidden_dims)
    momentum = None

    if criterion == "contrastive":
        loss_func = ContrastiveLoss()
    elif criterion == "ms":
        alpha = trial.suggest_int('alpha', 1, 10, log=True)
        beta = trial.suggest_int("beta", 20, 80)
        base = trial.suggest_float('base', 0.1, 1.0, log=True)
        loss_func = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)

    # Select optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Print current trial config and result
    print("\n==== Trial Summary ====")
    print(f"Embedding Dim: {embedding_dim}")
    print(f"Hidden Dims: {hidden_dims}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning Rate: {lr:.6f}")
    if momentum:
        print(f"Momentum: {momentum:.2f}")

    # Train
    results = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_func,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        patience=2,
        experiment_name='SiameseResNet',
        tuning_mode=True
    )

    objective_result = results['val_loss_history'][-1]

    return objective_result


def run_optuna_study(train_loader, val_loader, n_trials=50, study_name="siamese_study", criterion:str = "contrastive"):
    # Create a study with TPE sampler
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # We want to minimize validation loss
        storage=f"sqlite:///{study_name}.db",  # Save results to SQLite DB
        load_if_exists=True
    )

    # Wrap the objective with lambda to pass additional arguments
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, criterion=criterion),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Print and save results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save importance plot
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{study_name}_importance.html")

    # Save optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{study_name}_history.html")

    return study