from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(data_path="/data", batch_size=8):
    # Load the MNIST train and test datasets and save them under data_path
    mnist_train, mnist_test = get_mnist_data_sets(data_path)

    train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(mnist_test, batch_size=1000, shuffle=True)
    return train_loader, val_loader


def get_mnist_data_sets(data_path="/data"):
    mnist_train = datasets.MNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    mnist_test = datasets.MNIST(
        data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    return mnist_train, mnist_test
