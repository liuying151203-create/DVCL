# DVCL 实验代码

本仓库提供 DVCL 的可复现实验代码，用于统一生成主实验、鲁棒性实验和消融实验结果。论文正文不属于本公开仓库；方法说明、实验配置和复现步骤以本 README 及 `docs/` 中的公开文档为准。

## 设计原则

所有模型必须使用相同的实验产物链：

```text
原始图 → 固定数据划分 → 固定攻击图 → 模型训练 → JSON 结果
```

训练代码不得临时重新划分数据或重新生成攻击。划分随机种子、攻击随机种子和训练随机种子相互独立，并写入运行清单，以便公平比较和复现。

## 目录结构

```text
./
├── src/dvcl_bench/       # 实验协议、路径、清单和框架适配
├── configs/              # 主协议、复现协议、模型、攻击与消融配置
├── scripts/              # 稳定命令行入口
├── tests/                # 轻量实验契约测试
├── data/                 # 数据、划分和攻击产物，不纳入 Git
└── outputs/              # 结果、检查点和汇总，不纳入 Git
```

`src/dvcl_bench/` 负责统一实验协议与运行组织。本公开仓库不包含 HSeCo 作者源码或此前保存的兼容快照；HSeCo 基线只能通过依据已核实论文独立编写并验证的原生适配器接入。

## 环境安装

建议使用 Python 3.9–3.11，并根据本机 Torch 和 CUDA 版本安装匹配的 DGL 及可选 PyG 扩展：

```powershell
pip install -e .
```

GPU 依赖安装前请先检查 `requirements.txt`，避免 Torch、CUDA 和 DGL 版本不匹配。

## 单次实验

下面展示统一单次实验入口。当前 `legacy` 后端仅用于本地授权审计，不属于公开发行内容；公开版本应使用完成验证的 `native` 或 `openhgnn` 后端：

```powershell
python scripts/run_experiment.py `
  --model dvcl `
  --backend legacy `
  --dataset acm `
  --attack clean `
  --split-name seed_1 `
  --split-seed 1 `
  --attack-seed 1 `
  --train-seed 1 `
  --device cuda:0
```

添加 `--dry-run` 可以只检查最终命令，不启动训练。

## 批量实验

```powershell
python scripts/run_suite.py `
  --config configs/protocols/dvcl_main.yaml `
  --dry-run
```

确认 `data/` 下的数据、划分和攻击产物准备完成后，再移除 `--dry-run`。

消融实验配置位于 `configs/suites/dvcl_component_ablation.yaml`。消融实验应与完整 DVCL 使用相同划分、攻击、训练轮数、早停规则和调参预算，只改变被消融组件。

## 数据与攻击产物

统一使用顶层脚本准备实验产物：

```powershell
python scripts/prepare_dataset.py --dataset acm
python scripts/generate_split.py --dataset acm --seed 1 --protocol paper
python scripts/generate_attack.py --dataset acm --split paper_seed_1 --attack prbcd --attack-rate 5 --seed 1 --mode import --source-file PATH_TO_ATTACK.pt
python scripts/verify_attack.py --dataset acm --split paper_seed_1 --attack-path data/attacks/acm/prbcd/rate_5/seed_1/attack.pt
```

被攻击数据保存在 `data/attacks/`，固定划分保存在 `data/splits/`。每个模型读取同一份划分和攻击产物，不能在各自训练脚本中重复生成。

## 实验协议

项目区分三类协议：

1. `dvcl_main`：所有模型使用相同数据、划分、攻击、评估指标和调参预算；论文主表只使用这里的结果。
2. `paper_reproduction/*`：按基线作者的原始设置复核其论文结果；结果用于复现审计或附录。
3. `hgb_official`：可选的 HGB 官方数据与划分补充实验。

不能直接把基线论文中的数字当作统一协议下的主表结果。原论文数字可作为引用值单列；如果数据版本、划分、攻击或评估流程不同，应明确标注。

## 开源边界

本仓库不发布 HSeCo 原作者源码、无明确许可证的代码快照或由其直接改写的派生文件。HSeCo 的公开基线实现遵循独立重写原则：实现者只依据已核实的论文、补充材料和公开实验协议编写代码，并通过固定输入、指标和消融行为验证等效性。

HSeCo 的正式书目信息已通过 DOI `10.1145/3746252.3761343` 核实，但公开仓库尚未取得包含方法公式、伪代码和实验细节的论文正文，独立实现仍未完成。因此不能将内部兼容后端或其历史结果标记为公开 HSeCo 复现结果。具体要求见 `docs/hseco_reimplementation.md`。

## 基线接入

计划统一接入的基线包括：

- 通用异构图模型：HAN、HGT、MAGNN、HeCo、SimpleHGN；
- 鲁棒异构图模型：HSeCo、RoHe、HeteGuard；
- 无注意力鲁棒基线：FastRo-HGCN。

OpenHGNN 只作为模型实现来源。数据指纹、划分、攻击、调参预算、指标和输出清单仍由本项目控制。目前已保留 OpenHGNN 适配边界，并新增依据 HSeCo 论文公式编写的原生模型核心；数据与训练适配器尚未完成。在适配器和论文未说明的复现假设冻结前，不应把占位配置产生的内容作为论文结果。HSeCo 的论文实现规格见 `docs/hseco_paper_spec.md`。

## 结果要求

每次正式运行至少记录：

- 数据集及其指纹；
- 划分名称与划分随机种子；
- 攻击方法、扰动率与攻击随机种子；
- 模型配置与训练随机种子；
- 软件版本、运行命令、指标和检查点路径。

主实验应运行多个训练随机种子，报告均值和标准差。复现结果与原论文不一致时，优先核对数据版本、划分、攻击目标、指标计算、早停方式和超参数搜索预算，不应通过复制原论文数字掩盖协议差异。
