import torch
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from dataloaders import get_mnist_dataloaders
from neural_network import Net
from training import train, validate


def objective(options=None):
    # Initialize the best validation loss, which is the value to be minimized by the network
    best_val_loss = float("Inf")

    # Define hyperparameters
    lr = 0.001
    dropout = 0.3
    batch_size = 128
    print(f"Learning rate: {lr}")
    print(f"Dropout: {dropout}")
    print(f"Batch size: {batch_size}")

    # Use CUDA if GPU is available, else CPU
    use_cuda = options["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device {device}")

    # Obtain the MNIST train and validation loaders using a helper function
    train_loader, val_loader = get_mnist_dataloaders(options["data_path"], batch_size)

    # Initialize network
    model = Net(dropout=dropout).to(device)

    # Learning rate optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Network training & validation loop
    for epoch in range(0, options["epochs"]):
        avg_train_loss = train(
            options, model, device, train_loader, optimizer, epoch
        )
        avg_val_loss = validate(model, device, val_loader)

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss

        # Print intermediate validation & training loss
        print(f"Epoch {epoch + 1} of {options['epochs']} --- average train loss: {avg_train_loss} --- average validation loss: {avg_val_loss}")

        scheduler.step()

    # Return the best validation loss
    return best_val_loss


def main():
    # Experiment options
    options = {
        "epochs": 2,
        "use_cuda": True,
        "log_interval": 10,
        "data_path": "/data"
    }
    print("Options: ")
    for key, value in options.items():
        print(f"    {key}: {value}")

    # Run training and return the best achieved validation loss
    best_val_loss = objective(options)

    print("Finished training.")
    print(f"Best validation loss: {best_val_loss}")
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    main()
