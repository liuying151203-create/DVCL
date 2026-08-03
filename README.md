# DVCL 实验代码

本仓库用于在统一数据、划分、攻击和评估协议下运行 DVCL 与异构图鲁棒性基线。当前原生支持 ACM、DBLP、HSeCo 和 DVCL。

## 实验流程

```text
原始数据 → clean artifact → split artifact → attack artifact
        → 模型训练 → checkpoint + metrics + manifest
```

训练脚本只读取已冻结的 artifact。划分、攻击和训练分别使用独立随机种子，所有模型必须读取相同的划分和攻击图。

## 安装

建议使用 Python 3.9–3.11。CPU 验证环境：

```bash
pip install -r requirements-cpu.txt
python scripts/check_environment.py --profile cpu --smoke
```

CUDA 12.1 正式实验环境：

```bash
conda create -p .conda/dvcl-cu121-py311 python=3.11 pip -y
.conda/dvcl-cu121-py311/bin/python -m pip install -r requirements-cu121.txt
DVCL_GPU_ENV="$PWD/.conda/dvcl-cu121-py311" \
  source scripts/activate_gpu_env.sh
python scripts/check_environment.py --profile gpu --smoke
```

正式协议请求 CUDA 时不会静默回退 CPU。完整安装和环境审计说明见 `docs/environment.md`。

## 准备实验产物

```powershell
python scripts/prepare_dataset.py --dataset acm
python scripts/generate_split.py --dataset acm --seed 1 --protocol paper
```

生成 RND：

```powershell
python scripts/generate_attack.py --dataset acm --split paper_seed_1 --attack rnd --attack-rate 5 --seed 1
```

导入 PRBCD 或 HetePRBCD：

```powershell
python scripts/generate_attack.py --dataset acm --split paper_seed_1 --attack prbcd --attack-rate 5 --seed 1 --mode import --source-file PATH_TO_ATTACK.pt
```

验证攻击：

```powershell
python scripts/verify_attack.py --dataset acm --split paper_seed_1 --attack-path data/attacks/acm/prbcd/rate_5/seed_1/attack.pt
```

按协议一次性审计全部 clean、split 和 attack artifact：

```powershell
python scripts/check_protocol_inputs.py --config configs/protocols/dvcl_main.yaml
```

DBLP 全局实验的 `paper_seed_1` 使用与现有 PRBCD/HetePRBCD 源文件一致的
405/405/3247 划分。导入器会拒绝特征、标签或 split mask 不一致的攻击源。

旧 HSeCo artifact 可使用 `scripts/import_legacy_artifact.py` 转换为当前版本化格式。

## 运行实验

检查主实验矩阵：

```powershell
python scripts/run_suite.py --config configs/protocols/dvcl_main.yaml --dry-run
```

开始或继续实验：

```powershell
python scripts/run_suite.py --config configs/protocols/dvcl_main.yaml --continue-on-error
```

已完成的运行会自动跳过；使用 `--force` 可重新运行。单次实验入口为 `scripts/run_experiment.py`。

当前主协议包含：

- 数据集：ACM、DBLP；
- 模型：HSeCo、DVCL；
- 攻击：Clean、PRBCD、HetePRBCD；
- 扰动率：5%、10%、15%、20%、25%；
- 种子：固定 split seed 和 attack seed，运行 5 个 train seed；
- 指标：Accuracy、Micro-F1、Macro-F1。

其他基线将在相同 artifact 接口上逐个接入，不能直接混用原论文在不同划分或攻击协议下的数字。

## 消融实验

```powershell
python scripts/run_suite.py --config configs/suites/dvcl_component_ablation.yaml --dry-run
python scripts/run_suite.py --config configs/suites/dvcl_component_ablation.yaml
```

包含完整 DVCL、移除对比损失、仅拓扑视图和仅特征视图。各变体共用相同划分、攻击图和训练预算。

## 输出

每次运行保存在：

```text
outputs/runs/{protocol}/{dataset}/{model}/{variant}/{attack}/{rate}/{seeds}/
```

主要文件包括：

- `manifest.json`：输入文件哈希、Git 状态、配置和软件版本；
- `attack_verification.json`：攻击图检查结果；
- `history.csv`：逐轮训练和验证记录；
- `checkpoint.pt`：最佳模型；
- `metrics.json`：测试指标；
- `status.json`：运行、完成或失败状态。

汇总结果：

```powershell
python scripts/summarize_results.py
```

结果写入 `outputs/summaries/`，包括单次结果、均值标准差和跨攻击条件平均值。

## HSeCo 等效实现

HSeCo 实现参考论文和可获得源码中的数据处理、元路径转换、两级净化、对比损失、早停和 checkpoint 行为。论文复现协议可启用旧式 checkpoint 语义；统一主协议恢复完整语义模块和分类模块。详细说明见 `docs/hseco.md`。

## 后续开发路线

当前工程状态、golden 对照覆盖、正式环境建设、DBLP 产物准备、主实验和消融实验的分阶段计划见 `docs/development-roadmap.md`。

已完成的 ACM 主实验、攻击平均、组件消融和结果完整性审计见
`docs/acm-experiment-results.md`。
