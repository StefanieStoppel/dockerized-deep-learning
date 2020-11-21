import os
import mlflow
import optuna
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from mlflow import pytorch
from pprint import pformat
from urllib.parse import urlparse
from dataloaders import get_mnist_dataloaders
from neural_network import Net
from training import train, validate


def suggest_hyperparameters(trial):
    # Obtain the learning rate on a logarithmic scale
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    # Obtain the dropout ratio in a range from 0.0 to 0.9 with step size 0.1
    dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
    # Obtain the batch size (as power of 2)
    batch_size = 2 ** trial.suggest_int("batch_size_power", 5, 8, step=1)
    # Obtain the optimizer to use by name
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta"])

    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    # Log the obtained trial parameters using mlflow
    mlflow.log_params(trial.params)
    return lr, dropout, optimizer_name, batch_size


def objective(trial, experiment, options=None):
    # Initialize the best validation loss, which is the value to be minimized by the network
    best_val_loss = float("Inf")

    # Start mlflow run
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Use mlflow to log experiment options
        mlflow.log_params(options)

        # Get hyperparameter suggestions created by optuna
        lr, dropout, optimizer_name, batch_size = suggest_hyperparameters(trial)

        print(f"\n**************************")

        active_run = mlflow.active_run()
        print(f"Starting run {active_run.info.run_id} and trial {trial.number}")

        # Parse the active mlflow run's artifact_uri and convert it into a system path
        parsed_uri = urlparse(active_run.info.artifact_uri)

        artifact_path = os.path.abspath(
            os.path.join(parsed_uri.netloc, parsed_uri.path)
        )
        print(f"Artifact path for this run: {artifact_path}")

        # Use CUDA if GPU is available, else CPU
        use_cuda = options["use_cuda"] and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # Log mlflow device parameter
        mlflow.log_param("device", device)

        # Obtain the MNIST train and validation loaders using a helper function
        train_loader, val_loader = get_mnist_dataloaders()

        # Initialize network
        model = Net(dropout=dropout).to(device)

        # Pick an optimizer based on optuna's parameter suggestion
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        if optimizer_name == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

        # Network training & validation loop
        for epoch in range(0, options["epochs"]):
            avg_train_loss = train(
                options, model, device, train_loader, optimizer, epoch
            )
            avg_val_loss = validate(model, device, val_loader)

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            # Report intermediate objective value.
            trial.report(avg_val_loss, step=epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Log average train and test set loss for the current epoch using mlflow
            mlflow.log_metric("avg_train_losses", avg_train_loss, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            scheduler.step()

        # Save the final network model to the current mlflow run's directory
        if options["save_model"]:
            pytorch.save_model(model, f"{artifact_path}/mnist_model")

    # Return the best validation loss
    return best_val_loss
