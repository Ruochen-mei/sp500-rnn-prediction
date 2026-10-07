# S&P 500 Return Prediction with a Manual Vanilla RNN

使用历史 S&P 500 OHLCV 信息，学习预测下一交易日的 log return，并恢复为收盘价。这是一个 PyTorch 时间序列机器学习学习/实验项目，不是实际投资建议，也不代表可盈利的交易策略。

## 当前版本：Relative Features → Next-day Log Return

每个交易日构造五个相对特征（`epsilon = 1e-8`）：

| 特征 | 计算公式 |
|---|---|
| close_return | `log(Close_t / Close_(t-1))` |
| oc_return | `log(Close_t / Open_t)` |
| range | `(High_t - Low_t) / Close_(t-1)` |
| gap | `log(Open_t / Close_(t-1))` |
| volume_change | `log((Volume_t + epsilon) / (Volume_(t-1) + epsilon))` |

输入是过去 20 个交易日的五维特征：`X.shape = [B, 20, 5]`。目标是紧接窗口之后一天的 `log(Close_(t+1) / Close_t)`，`y.shape = [B, 1]`。目标列存储当天 return，由滑动窗口取下一行作为标签，无需再向前 shift。

`ManualVanillaRNN` 手写时间步循环，不使用 `nn.RNN`：

```text
h_t = tanh(W_xh(x_t) + W_hh(h_(t-1)))
prediction = output_layer(h_T)

W_xh: Linear(5, 64)
W_hh: Linear(64, 64, bias=False)
output_layer: Linear(64, 1)
```

训练使用 MSELoss、AdamW、学习率 1e-3、batch size 32、30 epochs、gradient clipping 1.0。支持 CUDA，不可用时使用 CPU。按最低 Validation Loss 保存最佳权重；当前 S&P 500 项目没有 early stopping。

## 数据与时间切分

将 CSV 放在 `data/SP500.csv`（注意大小写）。必须包含 `Date, Open, High, Low, Close, Volume`；日期应可解析，OHLCV 为数值。当前本地数据共 24532 行，覆盖 1927-12-30～2025-08-29。训练使用 1990 年起的数据。

原始数据的下载来源和再分发许可尚未在项目中记录，因此 `.gitignore` 默认排除该 CSV。公开发布前应补充实际来源、获取方式及再分发许可；项目不提供自动下载脚本，也不声称数据来自某一特定供应商。没有本地 CSV 时不能运行训练。

| 集合 | 当前实际日期范围 | 数据行数 | 窗口数 |
|---|---|---:|---:|
| Train | 1990-01-02～2020-12-14 | 7800 | 7780 |
| Validation | 2020-12-15～2023-04-21 | 591 | 591 |
| Test | 2023-04-24～2025-08-29 | 591 | 591 |

`config.py` 使用固定日期边界，运行时不按比例重切。`TRAIN_START` 为 1990-01-01；Test 从 2023-04-24 延续至 CSV 末尾，因此更换数据快照会改变测试集。

先排序，再使用当天与前一天数据构造特征。Feature scaler 和 target scaler 分别仅 fit Train。Validation 使用 Train 最后 20 天作为初始 context，Test 使用 Validation 最后 20 天；所有 DataLoader 均 `shuffle=False`。评估是逐日一步预测，后续测试窗口可以使用已经发生的真实历史，不是多步递归预测。

## 项目结构

```text
stock_rnn/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data.py
├── main.py
├── predict.py
├── utils.py
├── models/
│   ├── __init__.py
│   └── vanilla_rnn.py
├── data/SP500.csv                 # 自行准备，默认不提交
├── checkpoints/best_rnn.pt        # 训练生成，默认不提交
├── predictions.csv               # 当前 V2 实验结果
├── prediction_plot.png
├── explore.ipynb                  # 早期 V1 探索记录，不是当前训练入口
├── analyze_data.py
├── yearly_data_quality.csv
└── data_quality_by_year.png
```

`explore.ipynb` 保留早期 2010 起点、absolute OHLCV/Close、比例切分的学习记录；当前 V2 的实现以 Python 脚本和 `config.py` 为准。中国指数 Excel 不属于本项目，已排除提交。

## 安装

建议使用 Python 3.11（本次验证环境）。进入包含 `main.py` 的项目根目录后执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux 激活：

```bash
source .venv/bin/activate
```

然后安装：

```bash
python -m pip install -r requirements.txt
```

运行依赖仅包含 PyTorch、NumPy、pandas、scikit-learn、Matplotlib。若要交互式打开历史 notebook，可额外安装 `jupyterlab`；它不是训练/预测依赖。CUDA 是否可用取决于本机驱动及 PyTorch 安装。

本次已有环境版本：PyTorch 2.8.0+cu128、NumPy 2.1.2、pandas 2.3.2、scikit-learn 1.7.1、Matplotlib 3.10.6。requirements 使用围绕这些已验证版本的小版本范围，不是精确环境锁文件。

## 运行

所有命令必须在项目根目录执行，代码使用相对路径。

```bash
python -c "from data import load_data; load_data()"
python main.py
python predict.py
```

`main.py` 训练并保存 `checkpoints/best_rnn.pt`，重新加载最佳权重后输出 Test Loss。`predict.py` 使用该权重评估 Test，保存 `predictions.csv` 和 `prediction_plot.png` 并显示图像。必须使用当前 V2 训练的权重，旧 V1 权重虽然形状相同但语义不兼容。

预测价格为 `Previous_Close * exp(Predicted_Return)`；Naive baseline 为前一交易日真实 Close，即预测 return 为 0。CSV 包含 Date、Previous_Close、Actual_Close、Predicted_Close、Naive_Close、Actual_Return、Predicted_Return。

可选数据质量检查：

```bash
python analyze_data.py
```

运行训练/预测/分析会覆盖同名权重或结果文件；需要保留已有实验时先备份。无图形界面的环境可设置 `MPLBACKEND=Agg`。

## 当前保存的实验结果

以下指标直接根据当前 `predictions.csv` 的 591 个 Test 样本重新计算，保留现有实验结果，未用重新训练结果替换：

| 指标 | Vanilla RNN | Naive |
|---|---:|---:|
| Price MAE（指数点） | 34.8929 | 34.6483 |
| Price RMSE（指数点） | 51.9982 | 51.6628 |
| MAE improvement over Naive | -0.7059% | — |
| RMSE improvement over Naive | -0.6493% | — |
| Return MAE | 0.006645 | — |
| Return RMSE | 0.009817 | — |
| Directional Accuracy | 52.62% | — |

Improvement = `(Naive error - RNN error) / Naive error * 100%`；负值表示劣于 Naive。方向准确率按 `sign(predicted_return) == sign(actual_return)` 计算，包含零收益情况。当前模型的价格误差略高于 Naive，没有显示超越该基线的效果。

![Current V2 price prediction](prediction_plot.png)

当前目录没有保存完整训练日志，因此不把历史 V1 的 validation/test loss 当作 V2 训练结果。训练输出的 loss 是标准化 target 上的 MSE，与真实价格 RMSE 不同。

历史 V1（用户记录，1990 起点）直接用 absolute OHLCV 预测 Close：Validation Loss 0.068123、Test Loss 1.996452、Price MAE 901.31、Price RMSE 1049.87。V2 改为相对特征与 return target 后价格误差明显降低，但仍需与 Naive 比较；两个版本的 scaled loss 因目标含义不同不能直接比较。

当前实现未固定随机种子，也没有持久化 scaler（预测时根据相同 CSV 和配置重建）。重训数值可能不同；复现已有权重需要相同数据快照、配置和相容环境。这些实验未纳入交易成本或构建交易策略。

## 发布说明

建议提交源码、README、requirements、.gitignore、历史 notebook，以及用于展示的预测和数据质量结果。权重默认不提交；如需分享，可作为可选 Release 附件，并注明对应配置和数据快照。CSV 在确认来源和再分发许可前保持本地。项目暂未指定开源许可证，发布者应自行选择合适许可证。

本项目仅用于机器学习学习与实验，不构成投资建议。

## 运行验证记录

2026-10-07：在已有 Python 3.11 环境的隔离副本中，现有 checkpoint 预测、完整 30 轮训练、训练后的预测及数据质量分析均正常退出；历史 notebook 全部代码单元执行通过。验证结果没有覆盖项目原有实验文件。依赖安装仅完成已有环境的离线 dry-run 检查，未在全新虚拟环境中重新下载安装。
