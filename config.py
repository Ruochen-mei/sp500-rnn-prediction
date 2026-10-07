import torch

DATA_PATH = "data/SP500.csv"

# 只调整 TRAIN_START 即可比较不同训练历史长度
TRAIN_START = "1990-01-01"
TRAIN_END = "2020-12-14"
VAL_START = "2020-12-15"
VAL_END = "2023-04-21"
TEST_START = "2023-04-24"

WINDOW_SIZE = 20

FEATURE_COLS = [
    "close_return",
    "oc_return",
    "range",
    "gap",
    "volume_change"
]

TARGET_COL = "close_return"

INPUT_SIZE = 5

HIDDEN_SIZE = 64

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 1e-3

CLIP = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SAVE_PATH = "checkpoints/best_rnn.pt"
