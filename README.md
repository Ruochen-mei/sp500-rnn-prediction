# S&P 500 Prediction with Vanilla RNN
# 基于 Vanilla RNN 的 S&P 500 指数预测

## 中文

### 项目简介

本项目使用 PyTorch 实现 Vanilla Recurrent Neural Network（Vanilla RNN），利用 S&P 500 的历史市场数据预测下一交易日的收益率，并进一步还原下一交易日的收盘指数。

项目主要用于复现和理解 RNN 在时间序列任务中的完整工作流程。为了更直观地理解循环神经网络内部的计算过程，模型没有直接调用 PyTorch 的 `nn.RNN`，而是手动实现隐藏状态在时间序列上的迭代更新。

项目最初尝试直接使用历史 OHLCV 数据预测下一交易日的绝对收盘指数，但在跨时间段预测时出现了明显的泛化问题。因此，当前版本将任务调整为：

```text
历史 OHLCV
    ↓
构造相对变化特征
    ↓
过去 20 个交易日
    ↓
Manual Vanilla RNN
    ↓
预测下一交易日 Log Return
    ↓
还原下一交易日 Close
```

---

## 1. 预测任务

原始数据包含 S&P 500 每个交易日的：

```text
Open
High
Low
Close
Volume
```

为了降低长期价格水平变化对模型的影响，当前版本不直接使用绝对 OHLCV，而是构造 5 个相对特征：

| 特征 | 计算方式 |
|---|---|
| Close Return | `log(Close_t / Close_(t-1))` |
| Open-Close Return | `log(Close_t / Open_t)` |
| Daily Range | `(High_t - Low_t) / Close_(t-1)` |
| Opening Gap | `log(Open_t / Close_(t-1))` |
| Volume Change | `log((Volume_t + ε) / (Volume_(t-1) + ε))` |

其中：

```text
ε = 1e-8
```

每个样本使用过去 **20 个交易日**作为输入：

```text
X.shape = [batch_size, 20, 5]
```

预测目标为紧接着下一交易日的 log return：

```text
log(Close_(t+1) / Close_t)
```

因此：

```text
y.shape = [batch_size, 1]
```

预测得到 return 后，再还原为下一交易日的收盘指数：

```text
Predicted_Close_(t+1)
    = Close_t × exp(Predicted_Return_(t+1))
```

---

## 2. Manual Vanilla RNN

模型位于：

```text
models/vanilla_rnn.py
```

为了理解 Vanilla RNN 的内部计算，本项目没有使用 `nn.RNN`，而是手动实现时间步循环。

对于每一个时间步：

```text
h_t = tanh(W_xh(x_t) + W_hh(h_(t-1)))
```

模型依次处理 20 个交易日的数据，并不断更新 hidden state。

处理完整个序列后：

```text
prediction = output_layer(h_T)
```

模型结构为：

```text
Input size:   5
Hidden size:  64
Output size:  1

W_xh:         Linear(5, 64)
W_hh:         Linear(64, 64, bias=False)
Output layer: Linear(64, 1)
```

这一实现保留了 Vanilla RNN 最基本的循环结构，也便于后续与 LSTM、Transformer 等模型进行对比。

---

## 3. 数据与时间划分

本项目使用 S&P 500 日频 OHLCV 数据。

本地原始数据共有 24,532 条记录，时间范围为：

```text
1927-12-30 → 2025-08-29
```

当前实验使用 **1990 年以后**的数据，并严格按照时间顺序划分训练集、验证集和测试集：

| 数据集 | 时间范围 | 数据量 |
|---|---|---:|
| Train | 1990-01-02 → 2020-12-14 | 7800 |
| Validation | 2020-12-15 → 2023-04-21 | 591 |
| Test | 2023-04-24 → 2025-08-29 | 591 |

构造 20 日窗口后：

```text
Train:      (7780, 20, 5)
Validation: (591, 20, 5)
Test:       (591, 20, 5)
```

时间序列数据不进行随机切分。

Feature scaler 和 target scaler 都只使用 Train 数据进行拟合。Validation 可以使用 Train 末尾已经发生的 20 个交易日作为历史 context，Test 同理可以使用 Validation 末尾的历史数据。

整个过程中不会使用未来信息构造当前预测。

原始 S&P 500 CSV 不包含在仓库中。如需运行项目，需要自行准备：

```text
data/SP500.csv
```

并包含：

```text
Date, Open, High, Low, Close, Volume
```

---

## 4. 训练配置

当前 Vanilla RNN 使用：

| 参数 | 设置 |
|---|---:|
| Sequence Length | 20 |
| Input Size | 5 |
| Hidden Size | 64 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 1e-3 |
| Loss | MSELoss |
| Optimizer | AdamW |
| Gradient Clipping | 1.0 |

程序会自动检测 CUDA，在 GPU 可用时使用 GPU 训练。

训练过程中根据 Validation Loss 保存最佳模型：

```text
checkpoints/best_rnn.pt
```

模型权重由训练生成，不包含在当前 GitHub 仓库中。

---

## 5. 技术栈

项目主要使用：

- **Python**：项目主要开发语言
- **PyTorch**：Vanilla RNN、训练和推理
- **pandas**：时间序列数据读取与处理
- **NumPy**：数值计算
- **scikit-learn**：Feature / Target 标准化
- **Matplotlib**：结果可视化
- **Jupyter Notebook**：前期数据探索与实验

---

## 6. 项目结构

```text
stock_rnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config.py
├── data.py
├── main.py
├── predict.py
├── utils.py
│
├── models/
│   ├── __init__.py
│   └── vanilla_rnn.py
│
├── explore.ipynb
└── analyze_data.py
```

主要文件：

| 文件 | 作用 |
|---|---|
| `config.py` | 数据、模型及训练参数配置 |
| `data.py` | 特征构造、时间切分、标准化和序列生成 |
| `models/vanilla_rnn.py` | 手动实现 Vanilla RNN |
| `main.py` | 模型训练、验证和测试 |
| `predict.py` | 加载训练模型并进行完整预测评估 |
| `utils.py` | 训练和评估的公共函数 |
| `analyze_data.py` | 原始历史数据质量分析 |
| `explore.ipynb` | 前期探索及早期实验记录 |

原始数据、模型 checkpoint、预测 CSV 和运行生成的图片等文件不包含在仓库中。

---

## 7. 安装与运行

创建虚拟环境：

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

准备数据：

```text
data/SP500.csv
```

训练模型：

```bash
python main.py
```

训练完成后运行：

```bash
python predict.py
```

`predict.py` 会在 Test Set 上进行逐日一步预测，并将模型结果与 Naive Baseline 进行比较。

Naive Baseline 定义为：

```text
Predicted Close_(t+1) = Close_t
```

也就是假设下一交易日的 return 为 0。

---

## 8. 实验结果

最终模型在 **2023 年 4 月至 2025 年 8 月的 591 个测试样本**上进行评估。

| Metric | Vanilla RNN | Naive Baseline |
|---|---:|---:|
| Price MAE | 34.89 | 34.65 |
| Price RMSE | 52.00 | 51.66 |
| Return MAE | 0.006645 | — |
| Return RMSE | 0.009817 | — |
| Directional Accuracy | 52.62% | — |

### 从 Absolute Price 到 Return Prediction

项目的早期版本直接使用 absolute OHLCV 预测下一交易日的绝对 Close。

该版本在测试集上的结果约为：

```text
Price MAE:  901.31
Price RMSE: 1049.87
```

一个明显的问题是：随着 S&P 500 长期价格水平变化，后期测试数据的绝对数值范围与训练阶段存在较大差异，模型的泛化效果很差。

因此当前版本将输入改为相对变化特征，并将预测目标改成 next-day log return。

修改后：

```text
Price MAE:  34.89
Price RMSE: 52.00
```

相比直接预测绝对价格，模型在不同市场价格水平之间的泛化能力有了明显改善。

但与简单的 Naive Baseline 比较：

```text
Vanilla RNN MAE: 34.89
Naive MAE:       34.65

Vanilla RNN RMSE: 52.00
Naive RMSE:       51.66
```

当前 Vanilla RNN 仍然略差于 Naive Baseline。

模型的 Directional Accuracy 为：

```text
52.62%
```

虽然略高于 50%，但差距较小，仅凭当前测试结果还不足以说明模型具有稳定的方向预测能力。

因此，本次实验比较清楚地体现了两个问题：

1. **预测目标和数据表示方式很重要。**  
   使用 return 代替 absolute price，可以明显缓解长期价格尺度变化带来的泛化问题。

2. **解决价格尺度问题并不意味着解决了预测问题。**  
   下一交易日 return 本身具有较强噪声，仅依赖历史 OHLCV 的简单 Vanilla RNN 并没有明显超过 persistence baseline。

后续可以在保持数据、时间切分、预测目标和评估方式一致的情况下，继续比较：

```text
Vanilla RNN
     ↓
LSTM
     ↓
Transformer
```

从而进一步观察不同序列模型在同一预测任务上的表现。

---

# English

## Overview

This project implements a Vanilla Recurrent Neural Network (RNN) in PyTorch to predict the next-day return of the S&P 500 index using historical market data.

The project was developed as a reproduction and learning exercise for recurrent neural networks and financial time-series prediction. Instead of using PyTorch's built-in `nn.RNN`, the recurrent computation is implemented manually so that the hidden-state update and sequential computation remain explicit.

The first version directly predicted the next closing price from historical OHLCV values. After observing poor generalization across different market periods, the task was reformulated as:

```text
Historical OHLCV
       ↓
Relative Features
       ↓
Previous 20 Trading Days
       ↓
Manual Vanilla RNN
       ↓
Next-Day Log Return
       ↓
Next-Day Closing Price
```

---

## 1. Prediction Task

The original dataset contains daily:

```text
Open
High
Low
Close
Volume
```

Instead of directly using absolute OHLCV values, five relative features are constructed:

| Feature | Definition |
|---|---|
| Close Return | `log(Close_t / Close_(t-1))` |
| Open-Close Return | `log(Close_t / Open_t)` |
| Daily Range | `(High_t - Low_t) / Close_(t-1)` |
| Opening Gap | `log(Open_t / Close_(t-1))` |
| Volume Change | `log((Volume_t + ε) / (Volume_(t-1) + ε))` |

where:

```text
ε = 1e-8
```

Each sample contains the previous 20 trading days:

```text
X.shape = [batch_size, 20, 5]
```

The target is the next-day log return:

```text
log(Close_(t+1) / Close_t)
```

with:

```text
y.shape = [batch_size, 1]
```

The predicted return is converted back to the next closing price using:

```text
Predicted_Close_(t+1)
    = Close_t × exp(Predicted_Return_(t+1))
```

---

## 2. Manual Vanilla RNN

The model is implemented in:

```text
models/vanilla_rnn.py
```

The project does not use `nn.RNN`.

At each time step:

```text
h_t = tanh(W_xh(x_t) + W_hh(h_(t-1)))
```

After processing the complete 20-day sequence:

```text
prediction = output_layer(h_T)
```

Model dimensions:

```text
Input size:   5
Hidden size:  64
Output size:  1

W_xh:         Linear(5, 64)
W_hh:         Linear(64, 64, bias=False)
Output layer: Linear(64, 1)
```

The implementation keeps the recurrent computation transparent and also provides a simple baseline for later comparison with LSTM and Transformer architectures.

---

## 3. Dataset and Split

The local dataset contains 24,532 daily S&P 500 observations covering:

```text
1927-12-30 → 2025-08-29
```

The current experiment uses observations starting from 1990.

The data is split chronologically:

| Dataset | Period | Samples |
|---|---|---:|
| Train | 1990-01-02 → 2020-12-14 | 7800 |
| Validation | 2020-12-15 → 2023-04-21 | 591 |
| Test | 2023-04-24 → 2025-08-29 | 591 |

After constructing 20-day sequences:

```text
Train:      (7780, 20, 5)
Validation: (591, 20, 5)
Test:       (591, 20, 5)
```

The feature scaler and target scaler are fitted on the training set only.

The original dataset is not included in this repository. To reproduce the experiment, place a compatible dataset at:

```text
data/SP500.csv
```

with:

```text
Date, Open, High, Low, Close, Volume
```

---

## 4. Training Configuration

| Parameter | Value |
|---|---:|
| Sequence Length | 20 |
| Input Size | 5 |
| Hidden Size | 64 |
| Batch Size | 32 |
| Epochs | 30 |
| Learning Rate | 1e-3 |
| Loss | MSELoss |
| Optimizer | AdamW |
| Gradient Clipping | 1.0 |

CUDA is used automatically when available.

The checkpoint with the lowest validation loss is saved locally as:

```text
checkpoints/best_rnn.pt
```

Model checkpoints are not included in the repository.

---

## 5. Tech Stack

- **Python**
- **PyTorch**
- **pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Jupyter Notebook**

---

## 6. Project Structure

```text
stock_rnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config.py
├── data.py
├── main.py
├── predict.py
├── utils.py
│
├── models/
│   ├── __init__.py
│   └── vanilla_rnn.py
│
├── explore.ipynb
└── analyze_data.py
```

| File | Description |
|---|---|
| `config.py` | Dataset, model, and training configuration |
| `data.py` | Feature construction, chronological splitting, scaling, and sequence generation |
| `models/vanilla_rnn.py` | Manual Vanilla RNN implementation |
| `main.py` | Training, validation, checkpoint selection, and testing |
| `predict.py` | Prediction and evaluation on the test set |
| `utils.py` | Shared training and evaluation utilities |
| `analyze_data.py` | Historical dataset quality analysis |
| `explore.ipynb` | Early exploration and experiments |

Generated prediction files, figures, checkpoints, and the original dataset are not included in the repository.

---

## 7. Installation and Usage

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/SP500.csv
```

Train the model:

```bash
python main.py
```

Evaluate the trained model:

```bash
python predict.py
```

The prediction script compares the RNN with a naive persistence baseline:

```text
Predicted Close_(t+1) = Close_t
```

---

## 8. Results

The final model is evaluated on **591 test observations from April 2023 to August 2025**.

| Metric | Vanilla RNN | Naive Baseline |
|---|---:|---:|
| Price MAE | 34.89 | 34.65 |
| Price RMSE | 52.00 | 51.66 |
| Return MAE | 0.006645 | — |
| Return RMSE | 0.009817 | — |
| Directional Accuracy | 52.62% | — |

### From Absolute Price to Return Prediction

An earlier version directly predicted the absolute closing price from historical OHLCV values and obtained approximately:

```text
Price MAE:  901.31
Price RMSE: 1049.87
```

The model generalized poorly when the absolute level of the S&P 500 in the test period differed substantially from the training period.

After switching to relative features and next-day log-return prediction:

```text
Price MAE:  34.89
Price RMSE: 52.00
```

This substantially reduced the error associated with changes in the long-term price level.

However, the persistence baseline still performs slightly better:

```text
Vanilla RNN MAE: 34.89
Naive MAE:       34.65

Vanilla RNN RMSE: 52.00
Naive RMSE:       51.66
```

The model achieves a directional accuracy of:

```text
52.62%
```

which is slightly above 50%, but the difference is too small to establish a reliable predictive advantage from this experiment alone.

The experiment therefore highlights two observations:

1. **Target representation matters.**  
   Predicting returns instead of absolute prices substantially improves generalization across different market price levels.

2. **Better representation does not make next-day returns easy to predict.**  
   With historical OHLCV information alone, the Vanilla RNN does not outperform a simple persistence baseline.

A natural continuation is to keep the dataset, target, split, and evaluation procedure fixed while comparing:

```text
Vanilla RNN
     ↓
LSTM
     ↓
Transformer
```

This project is intended as an experiment in recurrent neural networks and financial time-series modeling rather than as a trading system.