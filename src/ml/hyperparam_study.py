import optuna
from optuna.samplers import TPESampler
from src.ml.resNet50 import SiameseResNet
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def objective(trial, criterion, train_loader, val_loader):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    margin = trial.suggest_float('margin', 0.5, 10.0)

    # Initialize model fresh for each trial to avoid parameter leakage
    model = SiameseResNet().to(device)

    
    optimizer = torch.optim.AdamW(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

    results = model.train_model_constructive(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=ContrastiveLoss(margin=margin),
        optimizer=optimizer,  # Pass the created optimizer
        num_epochs=5,
        device=device,
        patience=2,
        experiment_name='SiameseResNet',
        tuning_mode=True
    )

    return results['val_loss']


def run_optuna_study(train_loader, val_loader, n_trials=50, study_name="siamese_study"):
    # Create a study with TPE sampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # We want to minimize validation loss
        sampler=sampler,
        storage=f"sqlite:///{study_name}.db",  # Save results to SQLite DB
        load_if_exists=True
    )

    # Wrap the objective with lambda to pass additional arguments
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader),
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