import torch
import torch.nn.functional as F


def train(options, model, device, train_loader, optimizer, epoch):
    model.train()
    train_set_size = len(train_loader.dataset)
    num_batches = len(train_loader)
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % options["log_interval"] == 0:
            batch_size = len(data)
            print(
                f"Train Epoch: {epoch} [{batch_idx * batch_size}/{train_set_size} "
                f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.6f}"
            )
    avg_train_loss = train_loss / num_batches
    return avg_train_loss


def validate(model, device, test_loader):
    model.eval()
    test_set_size = len(test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_set_size

    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{test_set_size} "
        f"({100. * correct / test_set_size:.0f}%)\n"
    )
    return test_loss
