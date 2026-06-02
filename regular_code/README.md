# regular_code 代码说明与运行指南

`regular_code/` 是对当前项目中原本分散脚本的规范化整理版本，目标是把“单信源近场球面波到达角和距离估计”的完整流程整理成清晰、可复用、可运行的代码结构。

项目核心流程包括：

1. 生成 LSN 训练数据；
2. 训练 LSN，用于预测信源个数；
3. 训练 LCN，用于对协方差矩阵进行去噪和内插；
4. 将 LCN 输出的结构化结果转换为 `R1 / R2`；
5. 使用 MUSIC 谱搜索估计 `omega / phi`；
6. 将 `omega / phi` 转换为角度 `theta` 和距离 `r`。

## 目录结构

```text
regular_code/
├── README.md
├── __init__.py
├── config.py
│
├── data/
│   ├── __init__.py
│   ├── generate_lsn_data.py
│   └── lcn_data.py
│
├── models/
│   ├── __init__.py
│   ├── lsn.py
│   └── lcn.py
│
├── train/
│   ├── __init__.py
│   ├── train_lsn.py
│   └── train_lcn.py
│
├── inference/
│   ├── __init__.py
│   ├── predict_lsn.py
│   ├── predict_lcn_music.py
│   └── pure_music_pipeline.py
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_lsn_confusion.py
│   └── eval_lcn_rmse.py
│
└── utils/
    ├── __init__.py
    ├── array_signal.py
    ├── covariance.py
    ├── device.py
    └── music.py
```

## 运行前说明

所有命令建议在项目根目录下运行，也就是包含 `regular_code/`、`model/`、`data/`、`LSN_trained.pth`、`LCN_trained_omega_phi_A_normR1.pth` 的目录。

示例：

```bash
python3 -m regular_code.inference.predict_lsn
```

不要进入 `regular_code/` 内部直接运行单个 `.py` 文件，否则 Python 包导入路径可能不正确。

## 依赖环境

代码主要依赖：

```text
numpy
torch
matplotlib
```

如果环境中缺少依赖，可以安装：

```bash
pip install numpy torch matplotlib
```

如果你使用的是 Anaconda 环境，也可以用：

```bash
conda install numpy matplotlib
pip install torch
```

## 全局配置：`config.py`

`regular_code/config.py` 用于集中管理项目中的关键参数，避免多个脚本中重复写死参数。

主要包含三个部分：

### `PROJECT_ROOT`

项目根目录路径，用于定位数据集和模型权重。

### `SignalConfig`

基础信号参数：

- `N = 5`：阵元数；
- `snapshots = 1024`：快拍数；
- `ps = 1.0`：信号功率；
- `fc = 2e6`：载频；
- `wavelength = 3e8 / fc`：波长；
- `d = wavelength / 4`：阵元间距；
- `K = 1`：默认单信源。

### `LSNConfig`

LSN 相关配置：

- `SNR_min = -10.0`；
- `SNR_max = 15.0`；
- `num_samples = 500_000`；
- `dataset_path = data/LSNdataset.npz`；
- `checkpoint_path = LSN_trained.pth`。

### `LCNConfig`

LCN 和 MUSIC 相关配置：

- `SNR_dB = 10.0`；
- `SNR_dB_min = -10.0`；
- `SNR_dB_max = 15.0`；
- `phi_min = 0.01`；
- `phi_max = pi - 0.01`；
- `omega_grid_min = -pi / 2`；
- `omega_grid_max = pi / 2`；
- `omega_grid_num = 20001`；
- `phi_grid_min = 0.0`；
- `phi_grid_max = pi`；
- `phi_grid_num = 20001`；
- `normalize_R1 = True`；
- `diagonal_loading_alpha = 1e-3`；
- `checkpoint_path = LCN_trained_omega_phi_A_normR1.pth`。

## data 模块

`regular_code/data/` 负责数据生成和数据集封装。

### `data/generate_lsn_data.py`

该文件用于生成 LSN 的训练数据集。

对应原始脚本：

```text
data/single_source_LSN.py
```

主要功能：

- 随机生成信源数量 `K`，范围为 `1 ~ N-1`；
- 随机生成每个信源的角度和距离；
- 生成近场 steering vector；
- 构造接收信号矩阵 `X`；
- 加入噪声；
- 计算协方差矩阵 `R = XX^H / snapshots`；
- 将复协方差矩阵拆成实部和虚部，得到 `(N, N, 2)`；
- 保存为 `data/LSNdataset.npz`。

输出数据格式：

```text
X: (num_samples, 5, 5, 2)
K: (num_samples,)
```

运行命令：

```bash
python3 -m regular_code.data.generate_lsn_data
```

注意：默认会生成 `500000` 个样本，运行时间和内存占用较大。如果只是测试代码是否可运行，建议先在 `regular_code/config.py` 中临时把 `num_samples` 调小。

### `data/lcn_data.py`

该文件用于 LCN 的 on-the-fly 数据生成。

对应原始正确脚本：

```text
model/LCN_train_verision2.py
```

主要类：

```python
OmegaPhiDataset
```

该数据集不会提前生成 `.npz` 文件，而是在训练过程中实时随机采样：

1. 随机采样 `omega`；
2. 随机采样 `phi`；
3. 随机采样 SNR；
4. 根据 `omega / phi` 生成协方差矩阵 `R`；
5. 将 `R` 转换为 `(2, N, N)` 作为 LCN 输入；
6. 从 `R` 构造严格定义的 `R1 / R2`；
7. 对 `R1` 做相位归一化；
8. 构造 12 维目标向量 `u`。

LCN 输入格式：

```text
x: (2, 5, 5)
```

LCN 标签格式：

```text
u: (12,)
```

`u` 的含义是：

```text
R1 的 3 个复数元素：实部 3 个 + 虚部 3 个
R2 的 3 个复数元素：实部 3 个 + 虚部 3 个
总计 12 维
```

## models 模块

`regular_code/models/` 存放神经网络模型定义。

### `models/lsn.py`

该文件定义 LSN 网络。

对应原始脚本：

```text
model/LSN.py
```

主要类：

```python
LSN
```

功能：预测信源个数。

输入：

```text
(B, 2, 5, 5)
```

输出：

```text
(B, 4)
```

其中 4 个类别对应：

```text
class 0 -> K = 1
class 1 -> K = 2
class 2 -> K = 3
class 3 -> K = 4
```

网络结构包括：

- 普通卷积；
- BatchNorm；
- depthwise separable convolution；
- 多层全连接；
- Dropout；
- 最终分类输出。

### `models/lcn.py`

该文件定义 LCN 网络。

对应原始正确脚本：

```text
model/LCN_train_verision2.py
model/predict_LCN.py
```

主要类：

```python
LCN
```

功能：对协方差矩阵进行去噪和内插，输出用于 MUSIC 的结构化向量 `u`。

输入：

```text
(B, 2, 5, 5)
```

输出：

```text
(B, 12)
```

其中：

```text
12 = 4 * (N - 2), N = 5
```

网络结构包括：

- 2D 卷积；
- BatchNorm；
- depthwise separable convolution；
- `AdaptiveAvgPool2d(1)`；
- 全连接输出 12 维向量。

## train 模块

`regular_code/train/` 存放训练脚本。

### `train/train_lsn.py`

该文件用于训练 LSN。

对应原始脚本：

```text
model/LSN_train.py
```

主要流程：

1. 加载 `data/LSNdataset.npz`；
2. 读取：
   ```text
   X: (num_samples, 5, 5, 2)
   K: (num_samples,)
   ```
3. 将输入转换为 PyTorch 模型需要的格式：
   ```text
   (5, 5, 2) -> (2, 5, 5)
   ```
4. 将 `K = 1~4` 转为类别标签 `0~3`；
5. 使用 `CrossEntropyLoss` 训练；
6. 保存权重到：
   ```text
   LSN_trained.pth
   ```

运行命令：

```bash
python3 -m regular_code.train.train_lsn
```

注意：训练 LSN 需要先有数据集：

```bash
python3 -m regular_code.data.generate_lsn_data
```

如果项目根目录下已经有 `data/LSNdataset.npz`，则可以直接训练。

### `train/train_lcn.py`

该文件用于训练 LCN。

对应原始正确脚本：

```text
model/LCN_train_verision2.py
```

主要流程：

1. 创建 `OmegaPhiDataset`；
2. 随机生成 `omega / phi / SNR`；
3. 实时生成协方差矩阵 `R`；
4. 构造 LCN 输入 `(2, 5, 5)`；
5. 构造目标向量 `u`；
6. 使用 `MSELoss` 训练 LCN；
7. 保存权重到：
   ```text
   LCN_trained_omega_phi_A_normR1.pth
   ```

运行命令：

```bash
python3 -m regular_code.train.train_lcn
```

默认训练参数：

```text
num_train = 200000
num_val   = 20000
batch_size = 256
epochs = 20
lr = 1e-3
```

注意：这是较长时间训练任务。如果只是测试代码是否可运行，可以在代码中临时调用较小参数，例如：

```python
train_lcn(num_train=1000, num_val=200, batch_size=64, epochs=1)
```

## inference 模块

`regular_code/inference/` 存放推理和完整 pipeline 脚本。

### `inference/predict_lsn.py`

该文件用于测试 LSN 的信源数预测功能。

对应原始脚本：

```text
model/predict_LSN.py
```

主要流程：

1. 加载 `LSN_trained.pth`；
2. 随机生成 `K=1,2,3,4` 的测试样本；
3. 将协方差矩阵转换为 `(1, 2, 5, 5)`；
4. 输入 LSN；
5. 输出预测的信源数量和 softmax 概率。

运行命令：

```bash
python3 -m regular_code.inference.predict_lsn
```

成功运行时会输出类似：

```text
>>> 成功加载 LSN_trained.pth
真实 K = 1, 预测 K = 1
真实 K = 2, 预测 K = 2
真实 K = 3, 预测 K = 3
真实 K = 4, 预测 K = 4
```

### `inference/predict_lcn_music.py`

该文件是 LCN + MUSIC 的核心推理闭环。

对应原始正确脚本：

```text
model/predict_LCN.py
```

主要流程：

```text
给定 omega_true / phi_true
        ↓
生成协方差矩阵 R
        ↓
R -> two-channel tensor: (2, 5, 5)
        ↓
加载 LCN_trained_omega_phi_A_normR1.pth
        ↓
LCN 输出 u_pred: (12,)
        ↓
u_pred -> R1_hat / R2_hat
        ↓
对 R1_hat 做相位归一化
        ↓
MUSIC 搜索 omega
        ↓
MUSIC 搜索 phi
        ↓
输出估计结果
```

运行命令：

```bash
python3 -m regular_code.inference.predict_lcn_music
```

成功运行时会输出类似：

```text
===== Case #06 =====
====================================
[truth] omega=1.100000, phi=0.490000
[est  ] omega=1.094845 (raw=1.094845)
[est  ] phi  =0.503283 (raw=0.503283)
[u err] mean=..., max=...
```

该脚本可以验证最重要的闭环：

```text
LCN 去噪/内插 -> MUSIC 谱搜索 -> omega/phi 估计
```

### `inference/pure_music_pipeline.py`

该文件是不经过神经网络的纯 MUSIC baseline。

对应原始脚本：

```text
predict_pipeline.py
MUSIC.py
```

主要流程：

```text
给定 omega_true / phi_true
        ↓
生成协方差矩阵 R
        ↓
直接从 R 构造 R1 / R2
        ↓
MUSIC 搜索 omega / phi
        ↓
omega / phi 转换为 theta / r
        ↓
输出估计结果
```

运行命令：

```bash
python3 -m regular_code.inference.pure_music_pipeline
```

成功运行时会输出类似：

```text
=== Pure MUSIC estimation ===
theta_true(deg)=...
theta_hat(deg)=...
omega_true(rad)=...
omega_hat(rad)=...
phi_true(rad)=...
phi_hat(rad)=...
r_true(m)=...
r_hat(m)=...
```

该脚本用于验证 MUSIC 本身是否工作正常，也可以作为 LCN + MUSIC 的 baseline 对比。

## evaluation 模块

`regular_code/evaluation/` 存放评估脚本。

### `evaluation/eval_lsn_confusion.py`

该文件用于评估 LSN 的分类效果，并绘制混淆矩阵。

对应原始脚本：

```text
model/LSN_show.py
```

主要流程：

1. 加载 `data/LSNdataset.npz`；
2. 加载 `LSN_trained.pth`；
3. 按类别均衡抽样；
4. 预测每个样本的信源数量；
5. 统计混淆矩阵；
6. 保存并显示混淆矩阵图片。

运行命令：

```bash
python3 -m regular_code.evaluation.eval_lsn_confusion
```

默认每类抽样 5000 个样本。

如果想评估全部样本：

```bash
python3 -m regular_code.evaluation.eval_lsn_confusion --samples-per-class 0
```

如果想减少样本快速测试：

```bash
python3 -m regular_code.evaluation.eval_lsn_confusion --samples-per-class 100
```

默认输出图片：

```text
regular_code/evaluation/LSN_confusion_matrix.png
```

### `evaluation/eval_lcn_rmse.py`

该文件用于评估 LCN + MUSIC 在不同 SNR 下的角度和距离 RMSE。

对应原始脚本：

```text
model/visualization.py
test.py
```

主要流程：

1. 设置真实角度：
   ```text
   theta_true = 30 deg
   ```
2. 设置真实距离：
   ```text
   r_true = wavelength / 6
   ```
3. 转换为：
   ```text
   omega_true / phi_true
   ```
4. 遍历 SNR；
5. 每个 SNR 下进行 Monte Carlo 实验；
6. 每次实验生成协方差矩阵 `R`；
7. 使用 LCN 得到 `u_pred`；
8. 通过 MUSIC 得到 `omega_hat / phi_hat`；
9. 转换得到 `theta_hat / r_hat`；
10. 计算 RMSE；
11. 保存 `.npz` 结果和 RMSE 曲线图。

运行命令：

```bash
python3 -m regular_code.evaluation.eval_lcn_rmse
```

默认参数：

```text
SNR: -10 dB 到 15 dB
mc_runs: 1000
omega_grid_num: 4001
phi_grid_num: 4001
```

这是耗时较长的评估。如果只是快速测试，可以运行：

```bash
python3 -m regular_code.evaluation.eval_lcn_rmse --mc-runs 5 --snr-min 0 --snr-max 2 --snr-step 1 --omega-grid-num 401 --phi-grid-num 401
```

默认输出：

```text
rmse_results.npz
regular_code/evaluation/angle_rmse_vs_snr.png
regular_code/evaluation/range_rmse_vs_snr.png
```

## utils 模块

`regular_code/utils/` 存放公共工具函数。

### `utils/array_signal.py`

信号模型和物理参数转换工具。

主要函数：

```python
near_field_steering_geometric(theta, r, N, wavelength, d)
```

用于生成近场球面波 steering vector。

```python
generate_R_from_omega_phi(omega, phi, cfg, SNR_dB=None)
```

根据 `omega / phi` 生成单信源协方差矩阵 `R`。

```python
theta_r_to_omega_phi(theta_deg, r, fc, use_training_d=True)
```

将真实角度和距离转换为 `omega / phi`。

```python
omega_to_theta_deg(omega, wavelength, d)
```

将 `omega` 转换为角度 `theta`。

```python
theta_phi_to_r(theta_deg, phi, wavelength, d)
```

将 `theta / phi` 转换为距离 `r`。

### `utils/covariance.py`

协方差矩阵处理工具。

主要函数：

```python
R_to_tensor_2ch(R)
```

将复协方差矩阵转换为 LCN 输入：

```text
R: (5, 5) complex
-> (2, 5, 5) float32
```

```python
build_R1_R2_strict(R)
```

根据严格索引规则从完整协方差矩阵构造 `R1 / R2`。

```python
normalize_R1_phase(R1)
```

使用 `R1[0,0]` 对 `R1` 做相位和尺度归一化。

```python
build_u_A(R1, R2)
```

将 `R1 / R2` 编码为 12 维向量 `u`。

```python
u_A_to_R1_R2(u)
```

将 LCN 输出的 12 维向量还原为 `R1 / R2`。

```python
hermitianize(A)
diagonal_loading(R, alpha)
```

用于 MUSIC 前的矩阵稳定化处理。

### `utils/music.py`

MUSIC 谱搜索工具。

主要函数：

```python
music_2elem_single(R2x2, grid, cfg)
```

用于 2 元素单信源 MUSIC 搜索。

在 LCN + MUSIC 闭环中：

- 对 `R2` 搜索 `omega`；
- 对 `R1` 搜索 `phi`。

```python
music_spectrum(R, theta_grid, r_grid, K=1, wavelength=1.0)
```

用于二维 MUSIC baseline，直接搜索角度和距离。

```python
wrap_omega_to_physical_interval(omega_hat, wavelength, d)
```

将 `omega` 包裹回物理可行区间。

```python
wrap_phi_to_0_pi(phi_hat)
```

将 `phi` 包裹到 `[0, pi)`。

### `utils/device.py`

设备选择工具。

主要函数：

```python
get_device()
```

优先级：

```text
cuda -> mps -> cpu
```

也就是说：

- 如果有 NVIDIA GPU，使用 CUDA；
- 如果是 Apple Silicon 且支持 MPS，使用 MPS；
- 否则使用 CPU。

## 推荐运行顺序

如果你想从头完整运行整个项目，可以按下面顺序执行。

### 1. 生成 LSN 数据集

```bash
python3 -m regular_code.data.generate_lsn_data
```

如果已经存在：

```text
data/LSNdataset.npz
```

可以跳过这一步。

### 2. 训练 LSN

```bash
python3 -m regular_code.train.train_lsn
```

训练完成后会生成或覆盖：

```text
LSN_trained.pth
```

注意：如果你不想覆盖已有权重，请先备份原来的 `LSN_trained.pth`。

### 3. 测试 LSN 预测

```bash
python3 -m regular_code.inference.predict_lsn
```

### 4. 训练 LCN

```bash
python3 -m regular_code.train.train_lcn
```

训练完成后会生成或覆盖：

```text
LCN_trained_omega_phi_A_normR1.pth
```

注意：这是长时间训练任务。如果你不想覆盖已有权重，请先备份原来的权重文件。

### 5. 测试 LCN + MUSIC 闭环

```bash
python3 -m regular_code.inference.predict_lcn_music
```

这是最关键的推理验证脚本。

### 6. 测试纯 MUSIC baseline

```bash
python3 -m regular_code.inference.pure_music_pipeline
```

### 7. 评估 LSN 混淆矩阵

```bash
python3 -m regular_code.evaluation.eval_lsn_confusion
```

快速测试可以运行：

```bash
python3 -m regular_code.evaluation.eval_lsn_confusion --samples-per-class 100
```

### 8. 评估 LCN + MUSIC 的 RMSE 曲线

完整评估：

```bash
python3 -m regular_code.evaluation.eval_lcn_rmse
```

快速测试：

```bash
python3 -m regular_code.evaluation.eval_lcn_rmse --mc-runs 5 --snr-min 0 --snr-max 2 --snr-step 1 --omega-grid-num 401 --phi-grid-num 401
```

## 快速验证当前代码是否能形成闭环

如果你只是想确认 `regular_code/` 是否能跑通，不需要重新训练，可以直接使用当前项目根目录下已有的权重文件运行：

```bash
python3 -m regular_code.inference.predict_lsn
python3 -m regular_code.inference.predict_lcn_music
python3 -m regular_code.inference.pure_music_pipeline
```

这三条命令分别验证：

```text
LSN 信源数预测是否正常
LCN + MUSIC 推理闭环是否正常
纯 MUSIC baseline 是否正常
```

当前整理完成后，这三条命令已经验证过可以运行。

## 完整闭环说明

### LSN 闭环

```text
regular_code.data.generate_lsn_data
        ↓
生成 data/LSNdataset.npz
        ↓
regular_code.train.train_lsn
        ↓
生成 LSN_trained.pth
        ↓
regular_code.inference.predict_lsn
        ↓
预测信源数量 K
```

### LCN + MUSIC 闭环

```text
regular_code.data.lcn_data.OmegaPhiDataset
        ↓
实时生成 omega / phi / R / u
        ↓
regular_code.train.train_lcn
        ↓
生成 LCN_trained_omega_phi_A_normR1.pth
        ↓
regular_code.inference.predict_lcn_music
        ↓
LCN 输出 u_pred
        ↓
regular_code.utils.covariance.u_A_to_R1_R2
        ↓
得到 R1_hat / R2_hat
        ↓
regular_code.utils.music.music_2elem_single
        ↓
估计 omega_hat / phi_hat
        ↓
regular_code.utils.array_signal.omega_to_theta_deg
regular_code.utils.array_signal.theta_phi_to_r
        ↓
得到 theta_hat / r_hat
```

## 和原始散乱代码的对应关系

```text
原始文件                                  regular_code 中的新位置
────────────────────────────────────────────────────────────────────
data/single_source_LSN.py                  regular_code/data/generate_lsn_data.py
model/LSN.py                               regular_code/models/lsn.py
model/LSN_train.py                         regular_code/train/train_lsn.py
model/predict_LSN.py                       regular_code/inference/predict_lsn.py
model/LCN_train_verision2.py               regular_code/models/lcn.py
                                           regular_code/data/lcn_data.py
                                           regular_code/train/train_lcn.py
model/predict_LCN.py                       regular_code/inference/predict_lcn_music.py
predict_pipeline.py / MUSIC.py             regular_code/inference/pure_music_pipeline.py
model/LSN_show.py                          regular_code/evaluation/eval_lsn_confusion.py
model/visualization.py / test.py           regular_code/evaluation/eval_lcn_rmse.py
```

没有作为主流程迁移的旧代码：

```text
model/LCN.py
model/LCN_train.py
data/LCNdataset.npz 静态训练流程
tempCodeRunnerFile.py
model/__pycache__/
```

原因是当前确认的正确 LCN 训练和推理流程来自：

```text
model/LCN_train_verision2.py
model/predict_LCN.py
```

## 注意事项

1. 不建议直接删除原始散乱代码，除非确认 `regular_code/` 已完全满足后续使用需求。
2. 不建议随意覆盖已有 `.pth` 权重文件，特别是：
   ```text
   LSN_trained.pth
   LCN_trained_omega_phi_A_normR1.pth
   ```
3. LSN 数据生成和 LCN 训练都是较长任务，运行前建议确认时间和硬件资源。
4. 快速验证时优先运行 inference 脚本，不要直接重新训练。
5. 所有模块建议使用 `python3 -m ...` 的方式从项目根目录运行。
