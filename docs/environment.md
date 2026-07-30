# DVCL 实验环境

本文档定义 CPU 验证环境和 CUDA 12.1 正式实验环境。两个环境使用相同 Python 和核心依赖版本，但必须使用与设备匹配的 Torch 和 DGL wheel。

## 环境配置

| 配置 | 用途 | Torch | DGL | 正式结果 |
| --- | --- | --- | --- | --- |
| CPU | 单元测试、artifact、golden 和调试 | `2.1.2+cpu` | `1.1.3` | 否 |
| CUDA 12.1 | GPU pilot、主实验和消融 | `2.1.2+cu121` | `1.1.3+cu121` | 是 |

项目支持 Python 3.9–3.11，推荐 Python 3.11。

## CPU 验证环境

使用 Conda 创建隔离环境：

```bash
conda create -p .conda/dvcl-py311 python=3.11 pip -y
.conda/dvcl-py311/bin/python -m pip install -r requirements-cpu.txt
```

运行完整预检：

```bash
.conda/dvcl-py311/bin/python scripts/check_environment.py \
  --profile cpu \
  --smoke \
  --output outputs/environment/cpu-validation.json
```

CPU smoke 会执行：

- 版本检查；
- Torch、DGL 和 PyG 导入；
- DGL GraphConv 前向与反向；
- PyG GCNConv 前向与反向；
- 梯度有限值检查。

## CUDA 12.1 正式环境

在具有兼容 NVIDIA 驱动的 Linux GPU 主机上创建环境：

```bash
conda create -p .conda/dvcl-cu121-py311 python=3.11 pip -y
.conda/dvcl-cu121-py311/bin/python -m pip install -r requirements-cu121.txt
DVCL_GPU_ENV="$PWD/.conda/dvcl-cu121-py311" \
  source scripts/activate_gpu_env.sh
```

PyTorch 2.1.2 的 CUDA 12.1 安装方式见 [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)。DGL 1.1.3 的 Python 3.11 CUDA 12.1 wheel 位于 [DGL cu121 wheel index](https://data.dgl.ai/wheels/cu121/repo.html)。

运行正式预检：

```bash
python scripts/check_environment.py \
  --profile gpu \
  --smoke \
  --output outputs/environment/gpu-validation.json
```

只有报告中的顶层 `ok` 为 `true` 时，才允许启动正式实验。

`scripts/activate_gpu_env.sh` 会：

- 激活 `DVCL_GPU_ENV` 指定的 Conda 环境；
- 默认使用当前仓库的 `.conda/dvcl-cu121-py39`；
- 将 Torch 和 NVIDIA wheel 的动态库目录加入 `LD_LIBRARY_PATH`；
- 固定 `DGLBACKEND=pytorch`；
- 立即打印 Torch、DGL、PyG 和 CUDA 可用状态。

该脚本必须使用 `source` 执行，使环境变量保留在当前 shell。

## 设备策略

Runner 不再将不可用的 CUDA 请求静默回退到 CPU。

以下命令在 CUDA 不可用时必须失败，并在 `status.json` 中记录错误：

```bash
python scripts/run_experiment.py \
  --model hseco \
  --backend native \
  --dataset acm \
  --device cuda:0
```

CPU 验证必须显式指定：

```bash
python scripts/run_experiment.py \
  --protocol cpu_validation \
  --model hseco \
  --backend native \
  --dataset acm \
  --device cpu
```

不要把 CPU 验证结果写入正式主协议。

## Manifest 环境审计

manifest schema 版本为 2，并记录：

- Python 版本、解释器路径和实现；
- 操作系统、内核和机器架构；
- Torch、TorchVision、TorchAudio、DGL、PyG 和核心数据依赖版本；
- Torch 编译使用的 CUDA 版本；
- CUDA 是否可用；
- 可见 GPU 数量；
- GPU 名称、计算能力和显存；
- cuDNN 版本；
- DGL 后端；
- `CUDA_VISIBLE_DEVICES` 和 `DGLBACKEND`；
- Git commit、工作树状态和输入 artifact 哈希。

这些字段用于区分 CPU/GPU、定位跨设备数值差异，并审计正式结果是否来自指定环境。

## 正式实验启动门槛

启动 GPU pilot 前必须全部满足：

1. GPU 环境预检 `ok: true`；
2. `torch.cuda.is_available()` 为 `True`；
3. `torch.version.cuda` 为 `12.1`；
4. DGL 版本为 `1.1.3+cu121`；
5. Torch/DGL/PyG GPU smoke 通过；
6. 完整 PyTest 通过；
7. `scripts/check_contracts.py` 通过；
8. Git 工作树状态符合实验记录要求；
9. ACM GPU 单 epoch Runner 完成；
10. manifest 正确记录 GPU 和依赖版本。

## 当前机器状态

当前开发机没有可用 NVIDIA 驱动，因此：

- CPU 环境预检和真实反向 smoke 已通过；
- 已创建项目内 GPU 环境 `.conda/dvcl-cu121-py39`；
- 该环境使用 Python 3.9.25、Torch 2.1.2+cu121、DGL 1.1.3+cu121 和 PyG 2.5.3；
- CUDA 依赖和 DGL 动态库导入已通过；
- CUDA 请求失败策略可以验证；
- Python 3.11 GPU 环境目录 `.conda/dvcl-cu121-py311` 已初始化，但大体积 Torch wheel 尚未完成下载；
- GPU 正向、反向及 ACM 单 epoch smoke 必须在目标 GPU 主机完成。

当前机器可执行以下命令检查已安装环境：

```bash
source scripts/activate_gpu_env.sh
python scripts/check_environment.py --profile gpu
```

由于没有可用 NVIDIA 驱动，报告应只在设备可用性检查处失败；这不代表 CUDA wheel 安装失败。接入目标 GPU 后必须重新执行带 `--smoke` 的完整预检。
