# DVCL 实验代码

本仓库用于在统一数据、划分、攻击和评估协议下运行 DVCL 与异构图鲁棒性基线。当前支持 ACM、DBLP、AMiner 数据协议，以及 HSeCo、DVCL、HAN、HeteroSAGE、RoHe、HeteroGuard、FastRoHGCN、HGT、MAGNN、HeCo 和 SimpleHGN。

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

HGT、MAGNN、HeCo 和 SimpleHGN 使用固定 OpenHGNN 后端：

```bash
bash scripts/install_openhgnn_backend.sh
python scripts/check_openhgnn_backend.py
```

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

HAN 和 HeteroSAGE 基线使用独立的 220 次统一协议矩阵：

```powershell
python scripts/run_suite.py --config configs/protocols/baseline_main.yaml --dry-run
python scripts/run_suite.py --config configs/protocols/baseline_main.yaml --continue-on-error
```

具体模型语义和超参数见 `docs/baselines.md`。不能直接混用原论文在不同划分或
攻击协议下的数字。

HAN 和 HeteroSAGE 的 220 次矩阵已完成，当前结果及审计状态见
`docs/baselines.md`。

新增鲁棒基线、OpenHGNN 通用基线、RND、HG Baseline 目标逃逸和攻击因子对照使用独立 suite，执行顺序、
矩阵规模和 AMiner 数据要求见 `docs/experiment-expansion.md`。

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

从 `dvcl_main` 的 360 次主实验与消融运行生成论文总表：

```powershell
python scripts/generate_paper_tables.py
python scripts/generate_paper_tables.py --check
```

生成多攻击种子统计、显著性检验、下降幅度、平均排名和最终论文图表：

```powershell
python scripts/analyze_paper_results.py
```

最终表格只报告 Micro-F1。完整结果见 `docs/final-experiment-results.md`、
`docs/aminer-experiment-results.md` 和 `docs/target-evasion-results.md`。

## HSeCo 等效实现

HSeCo 实现参考论文和可获得源码中的数据处理、元路径转换、两级净化、对比损失、早停和 checkpoint 行为。论文复现协议可启用旧式 checkpoint 语义；统一主协议恢复完整语义模块和分类模块。详细说明见 `docs/hseco.md`。

批量 Golden 对照使用 `configs/golden/hseco_dvcl.yaml`，执行和审计规则见
`docs/golden.md`。

## 后续开发路线

当前工程状态、golden 对照覆盖、正式环境建设、DBLP 产物准备、主实验和消融实验的分阶段计划见 `docs/development-roadmap.md`。

已完成的 ACM 主实验、攻击平均、组件消融和结果完整性审计见
`docs/acm-experiment-results.md`。

已完成的 DBLP 主实验、攻击平均和结果分析见
`docs/dblp-experiment-results.md`。

跨数据集主实验、攻击平均与 ACM 消融论文表见
`docs/paper-experiment-tables.md`。

最终环境、Git、协议、artifact 和结果哈希冻结流程见
`docs/reproducibility.md`。
