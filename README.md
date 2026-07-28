# DVCL 实验代码

本仓库用于在统一协议下运行 DVCL、异构图基线和鲁棒性实验。论文源码由独立仓库管理，不包含在本仓库中。

## 实验原则

所有模型共用以下实验产物：

```text
原始数据 → 固定划分 → 固定攻击图 → 模型训练 → JSON 结果
```

划分、攻击和训练分别使用独立随机种子。主表只比较在相同数据版本、划分、攻击、指标和调参预算下重新运行的结果；基线论文中的原始数字仅用于复现核对。

## 目录

```text
src/dvcl_bench/   实验协议、模型和适配器
configs/          模型、主实验、复现与消融配置
scripts/          数据准备、攻击生成和实验入口
tests/            实验契约与模型测试
data/             数据、划分和攻击产物（不提交）
outputs/          日志、检查点和结果（不提交）
```

## 安装

建议使用 Python 3.9–3.11，并根据 CUDA 版本安装匹配的 PyTorch 和 DGL：

```powershell
pip install -e .
```

## 数据、划分与攻击

```powershell
python scripts/prepare_dataset.py --dataset acm
python scripts/generate_split.py --dataset acm --seed 1 --protocol paper
python scripts/generate_attack.py --dataset acm --split paper_seed_1 --attack prbcd --attack-rate 5 --seed 1 --mode import --source-file PATH_TO_ATTACK.pt
python scripts/verify_attack.py --dataset acm --split paper_seed_1 --attack-path data/attacks/acm/prbcd/rate_5/seed_1/attack.pt
```

固定划分保存在 `data/splits/`，攻击图保存在 `data/attacks/`。模型训练只能读取这些产物，不能在训练脚本中重新划分或攻击。

## 运行实验

先检查批量运行计划：

```powershell
python scripts/run_suite.py --config configs/protocols/dvcl_main.yaml --dry-run
```

准备好数据和适配器后，移除 `--dry-run` 开始运行。当前主协议配置见 `configs/protocols/dvcl_main.yaml`。

## 实验设置

- 数据集：当前主协议使用 ACM、DBLP；AMiner 可在完成数据适配后加入。
- 中毒攻击：Clean、PRBCD、HetePRBCD；扰动率为 5%、10%、15%、20%、25%。
- 训练：每个设置运行 5 个训练随机种子，报告均值和标准差。
- 指标：节点分类指标必须由统一评估器计算。
- 基线候选：HAN、HGT、MAGNN、HeCo、SimpleHGN、RoHe、HeteroGuard、FastRo-HGCN、HSeCo。

基线可以接入有许可证的官方实现、OpenHGNN 实现，或根据论文与可获得源码完成的等效实现；无论来源如何，都必须接入本项目的数据、划分、攻击和评估接口后重新训练。

## 消融实验

消融配置位于 `configs/suites/dvcl_component_ablation.yaml`，包含：

- `full`：完整 DVCL；
- `no_cl`：移除对比学习；
- `topology_only`：仅拓扑视图；
- `feature_only`：仅特征视图。

各变体必须使用相同划分、攻击图、训练轮数、早停规则和调参预算。配置当前覆盖 Clean、PRBCD 与 HetePRBCD，并在 0%、5%、15%、25% 的代表性扰动率上运行 5 个训练随机种子。

## HSeCo 基线

HSeCo 采用等效实现：可参考论文和可获得源码中的模型结构、数据处理、接口与训练流程，并补全未公开部分。实现应明确记录参考版本、必要改动和复现假设，不能标记为作者官方实现。方法、攻击设置与当前代码状态见 `docs/hseco.md`。

## 结果记录

正式结果至少保存数据指纹、划分种子、攻击参数、训练种子、模型配置、运行命令、软件版本、指标和检查点路径。复现数字与论文不一致时，优先检查数据版本、划分、攻击目标、指标、早停和超参数预算。
