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
                f"({100. * batch_idx / num_batches:.0f}%)]\tTraining loss: {loss.item():.6f}"
            )
    avg_train_loss = train_loss / num_batches
    return avg_train_loss


def validate(model, device, val_loader):
    model.eval()
    val_set_size = len(val_loader.dataset)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= val_set_size

    print(
        f"Validation set -- Average loss: {val_loss:.4f}, Accuracy: {correct}/{val_set_size} "
        f"({100. * correct / val_set_size:.0f}%)\n"
    )
    return val_loss
