import torch

def evaluate(
    model,
    data_loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss
