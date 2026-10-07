import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from config import (
    DATA_PATH,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    WINDOW_SIZE,
    FEATURE_COLS,
    TARGET_COL,
    BATCH_SIZE
)

def create_sequences(
    X,
    y,
    window_size
):

    X_seq = []
    y_seq = []

    for i in range(
        len(X) - window_size
    ):

        X_window = X[
            i:i + window_size
        ]

        # 窗口后紧接着的下一交易日 log return

        y_target = y[
            i + window_size
        ]

        X_seq.append(X_window)
        y_seq.append(y_target)

    return (
        np.array(X_seq),
        np.array(y_seq)
    )

def load_data(return_metadata=False):

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    epsilon = 1e-8
    df["Previous_Close"] = df["Close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["close_return"] = np.log(df["Close"] / df["Previous_Close"])
        df["oc_return"] = np.log(df["Close"] / df["Open"])
        df["range"] = (df["High"] - df["Low"]) / df["Previous_Close"]
        df["gap"] = np.log(df["Open"] / df["Previous_Close"])
        df["volume_change"] = np.log(
            (df["Volume"] + epsilon) / (df["Volume"].shift(1) + epsilon)
        )

    df = df.iloc[1:].copy()
    df = df[df["Date"] >= TRAIN_START].copy()
    if not np.isfinite(df[FEATURE_COLS].to_numpy()).all():
        raise ValueError("Relative features contain NaN or inf.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError(
            "Dates are not sorted."
        )

    if df[FEATURE_COLS].isna().any().any():
        raise ValueError(
            "Features contain missing values."
        )

    if df[TARGET_COL].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    train_df = df[
        df["Date"].between(TRAIN_START, TRAIN_END)
    ].copy()

    val_df = df[
        df["Date"].between(VAL_START, VAL_END)
    ].copy()

    test_df = df[
        df["Date"] >= TEST_START
    ].copy()

    print(
        f"Total samples: {len(df)}"
    )

    print(
        f"Train: {len(train_df)} | "
        f"{train_df['Date'].min().date()} "
        f"-> "
        f"{train_df['Date'].max().date()}"
    )

    print(
        f"Validation: {len(val_df)} | "
        f"{val_df['Date'].min().date()} "
        f"-> "
        f"{val_df['Date'].max().date()}"
    )

    print(
        f"Test: {len(test_df)} | "
        f"{test_df['Date'].min().date()} "
        f"-> "
        f"{test_df['Date'].max().date()}"
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 只能在 Train 上 fit
    feature_scaler.fit(
        train_df[FEATURE_COLS]
    )

    target_scaler.fit(
        train_df[[TARGET_COL]]
    )

    X_train = feature_scaler.transform(
        train_df[FEATURE_COLS]
    )

    X_val = feature_scaler.transform(
        val_df[FEATURE_COLS]
    )

    X_test = feature_scaler.transform(
        test_df[FEATURE_COLS]
    )

    y_train = target_scaler.transform(
        train_df[[TARGET_COL]]
    )

    y_val = target_scaler.transform(
        val_df[[TARGET_COL]]
    )

    y_test = target_scaler.transform(
        test_df[[TARGET_COL]]
    )

    X_train_seq, y_train_seq = (
        create_sequences(
            X_train,
            y_train,
            WINDOW_SIZE
        )
    )

    X_val_context = np.concatenate(
        [
            X_train[-WINDOW_SIZE:],
            X_val
        ],
        axis=0
    )

    y_val_context = np.concatenate(
        [
            y_train[-WINDOW_SIZE:],
            y_val
        ],
        axis=0
    )

    X_val_seq, y_val_seq = (
        create_sequences(
            X_val_context,
            y_val_context,
            WINDOW_SIZE
        )
    )

    X_test_context = np.concatenate(
        [
            X_val[-WINDOW_SIZE:],
            X_test
        ],
        axis=0
    )

    y_test_context = np.concatenate(
        [
            y_val[-WINDOW_SIZE:],
            y_test
        ],
        axis=0
    )

    X_test_seq, y_test_seq = (
        create_sequences(
            X_test_context,
            y_test_context,
            WINDOW_SIZE
        )
    )

    print(
        "Train sequences:",
        X_train_seq.shape,
        y_train_seq.shape
    )

    print(
        "Validation sequences:",
        X_val_seq.shape,
        y_val_seq.shape
    )

    print(
        "Test sequences:",
        X_test_seq.shape,
        y_test_seq.shape
    )

    X_train_tensor = torch.tensor(
        X_train_seq,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train_seq,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val_seq,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val_seq,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test_seq,
        dtype=torch.float32
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    result = (
        train_loader,
        val_loader,
        test_loader,
        feature_scaler,
        target_scaler
    )

    if return_metadata:
        test_metadata = test_df[
            ["Date", "Previous_Close", "Close", TARGET_COL]
        ].rename(columns={
            "Close": "Actual_Close",
            TARGET_COL: "Actual_Return"
        }).reset_index(drop=True)
        return (*result, test_metadata)

    return result
