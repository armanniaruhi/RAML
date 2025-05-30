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

    lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    model = SiameseResNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Initialize model

    if criterion == "contrastive":
        loss_func = ContrastiveLoss()
    elif criterion == "ms":
        alpha = trial.suggest_int('alpha', 1, 10, log=True)
        beta = trial.suggest_int("beta", 20, 80)
        base = trial.suggest_float('base', 0.1, 1.0, log=True)
        loss_func = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)


    # Print current trial config and result
    print("\n==== Trial Summary ====")
    print(f"Learning Rate: {lr:.6f}")

    # Train
    results = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=loss_func,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        patience=3,
        experiment_name='SiameseResNet',
        scheduler_type="cosine"
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