import os
import time

import torch
import torch.nn as nn

from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    EPOCHS,
    LEARNING_RATE,
    CLIP,
    DEVICE,
    SAVE_PATH
)

from data import load_data

from models.vanilla_rnn import (
    ManualVanillaRNN
)

from utils import evaluate

(
    train_loader,
    val_loader,
    test_loader,
    feature_scaler,
    target_scaler
) = load_data()

X_batch, y_batch = next(
    iter(train_loader)
)

print()
print(
    "X batch shape:",
    X_batch.shape
)

print(
    "y batch shape:",
    y_batch.shape
)

print()

model = ManualVanillaRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

print(model)
print()

print(
    "Device:",
    DEVICE
)

print()

X_test_batch = X_batch.to(
    DEVICE
)

with torch.no_grad():

    test_prediction = model(
        X_test_batch
    )

print(
    "Model input:",
    X_test_batch.shape
)

print(
    "Model output:",
    test_prediction.shape
)

print(
    "Target:",
    y_batch.shape
)

print()

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

os.makedirs(
    os.path.dirname(SAVE_PATH),
    exist_ok=True
)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    start_time = time.time()

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (
        X_batch,
        y_batch
    ) in enumerate(train_loader):

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        predictions = model(
            X_batch
        )

        loss = criterion(
            predictions,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    train_loss = (
        total_loss
        / total_samples
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    epoch_time = (
        time.time()
        - start_time
    )

    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss "
        f"{train_loss:.6f} | "
        f"Val Loss "
        f"{val_loss:.6f} | "
        f"Time "
        f"{epoch_time:.2f}s"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            SAVE_PATH
        )

        print(
            "  -> Best model saved"
        )

model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=DEVICE
    )
)

test_loss = evaluate(
    model,
    test_loader,
    criterion,
    DEVICE
)

print()
print("=========================")
print("Training Finished")
print("=========================")

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Test Loss: "
    f"{test_loss:.6f}"
)
