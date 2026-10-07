import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from config import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    DEVICE,
    SAVE_PATH
)

from data import load_data
from models.vanilla_rnn import ManualVanillaRNN

(
    train_loader,
    val_loader,
    test_loader,
    feature_scaler,
    target_scaler,
    test_metadata
) = load_data(return_metadata=True)

model = ManualVanillaRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

model.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=DEVICE
    )
)

model.eval()

print()
print("Loaded model:", SAVE_PATH)
print("Device:", DEVICE)

predictions_scaled = []
targets_scaled = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_device = X_batch.to(DEVICE)

        predictions = model(X_device)

        predictions_scaled.append(
            predictions.cpu().numpy()
        )

        targets_scaled.append(
            y_batch.numpy()
        )

predictions_scaled = np.concatenate(
    predictions_scaled,
    axis=0
)

targets_scaled = np.concatenate(
    targets_scaled,
    axis=0
)

print()
print(
    "Prediction shape:",
    predictions_scaled.shape
)

print(
    "Target shape:",
    targets_scaled.shape
)

predicted_returns = target_scaler.inverse_transform(
    predictions_scaled
).reshape(-1)

actual_returns = target_scaler.inverse_transform(
    targets_scaled
).reshape(-1)

np.testing.assert_allclose(
    actual_returns, test_metadata["Actual_Return"].to_numpy(),
    rtol=1e-5, atol=1e-8
)
actual_returns = test_metadata["Actual_Return"].to_numpy()
previous_close = test_metadata["Previous_Close"].to_numpy()
predictions = previous_close * np.exp(predicted_returns)
targets = test_metadata["Actual_Close"].to_numpy()
naive_predictions = previous_close.copy()

return_mae = np.mean(np.abs(predicted_returns - actual_returns))
return_rmse = np.sqrt(np.mean((predicted_returns - actual_returns) ** 2))
directional_accuracy = np.mean(
    np.sign(predicted_returns) == np.sign(actual_returns)
)

rnn_mae = np.mean(
    np.abs(
        predictions - targets
    )
)

rnn_rmse = np.sqrt(
    np.mean(
        (predictions - targets) ** 2
    )
)

naive_mae = np.mean(
    np.abs(
        naive_predictions - targets
    )
)

naive_rmse = np.sqrt(
    np.mean(
        (naive_predictions - targets) ** 2
    )
)

mae_improvement = (
    (naive_mae - rnn_mae)
    / naive_mae
    * 100
)

rmse_improvement = (
    (naive_rmse - rnn_rmse)
    / naive_rmse
    * 100
)

print()
print("=" * 55)
print("Test Results")
print("=" * 55)

print(
    f"Price MAE:          {rnn_mae:.2f}"
)

print(
    f"Price RMSE:         {rnn_rmse:.2f}"
)

print()

print(
    f"Naive MAE:        {naive_mae:.2f}"
)

print(
    f"Naive RMSE:       {naive_rmse:.2f}"
)

print()

print(
    f"MAE improvement over naive:  "
    f"{mae_improvement:.2f}%"
)

print(
    f"RMSE improvement over naive: "
    f"{rmse_improvement:.2f}%"
)

print(f"Return MAE:       {return_mae:.6f}")
print(f"Return RMSE:      {return_rmse:.6f}")
print(f"Directional Accuracy: {directional_accuracy:.2%}")

results = pd.DataFrame({
    "Date": test_metadata["Date"],
    "Previous_Close": previous_close,
    "Actual_Close": targets,
    "Predicted_Close": predictions,
    "Naive_Close": naive_predictions,
    "Actual_Return": actual_returns,
    "Predicted_Return": predicted_returns
})

print()
print("=" * 55)
print("First 10 Test Predictions")
print("=" * 55)

print(
    results.head(10).to_string(
        index=False
    )
)

results.to_csv(
    "predictions.csv",
    index=False
)

print()
print(
    "Saved predictions to predictions.csv"
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    results["Date"], results["Actual_Close"],
    label="Actual"
)

plt.plot(
    results["Date"], results["Predicted_Close"],
    label="Vanilla RNN"
)

plt.plot(
    results["Date"], results["Naive_Close"],
    label="Naive Baseline",
    alpha=0.7
)

plt.xlabel("Date")
plt.ylabel("S&P 500 Close")

plt.title(
    "S&P 500 Test Prediction (Return + Vanilla RNN)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "prediction_plot.png",
    dpi=150
)

print(
    "Saved plot to prediction_plot.png"
)

plt.show()
