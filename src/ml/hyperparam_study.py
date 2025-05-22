import optuna
from src.ml.resNet50 import SiameseResNet

def objective(trial, train_loader, val_loader, device='cuda'):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use for training

    Returns:
        Validation loss
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 20)

    # Create model
    model = SiameseResNet()

    # Train model and get validation loss
    val_loss = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        device=device
    )

    return val_loss

def optimize_hyperparameters(train_loader, val_loader, n_trials=50, device='cuda'):
    """
    Run Optuna optimization for the Siamese network.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        n_trials: Number of optimization trials
        device: Device to use for training

    Returns:
        study: Optuna study object with optimization results
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader, device),
        n_trials=n_trials,
        timeout=3600
    )

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study