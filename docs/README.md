# DVCL 文档导航

本目录按“主结果—专项附录—协议审计—工程历史”分层。若不同文档的数字或结论表述不一致，按下列优先级解释。

## 1. 必读入口

1. `final-experiment-results.md`：唯一论文结果主入口，包含实验设置、主表、统计检验、消融、异常结果审计和结论边界。
2. `next-experiment-plan.md`：当前有效的后续开发顺序、正式矩阵和验收标准。
3. `reproducibility.md`：环境、Git、协议、artifact 与结果哈希冻结及复现命令。
4. `../README.md`：项目安装、数据准备和命令入口。

## 2. 结果附录

| 文档 | 内容 |
|---|---|
| `acm-experiment-results.md` | ACM 单攻击种子主实验与消融明细 |
| `dblp-experiment-results.md` | DBLP 单攻击种子主实验明细 |
| `aminer-experiment-results.md` | AMiner 十一模型完整结果与攻击有效性审计 |
| `target-evasion-results.md` | ACM、DBLP、AMiner 的 HG 迁移攻击与 11 模型自适应目标逃逸结果 |
| `dvcl-view-diagnosis-results.md` | DVCL 五种视图模式的单种子失效诊断与阶段 E 决策 |
| `dvcl-stage-e-results.md` | 阶段 E 两个可靠性门控候选的统一结果、验收结论与模型冻结决定 |
| `dvcl-reliability-gate-results.md` | 未增强可靠性门控的自动生成明细 |
| `dvcl-reliability-gate-augmentation-results.md` | 训练时结构增强门控的自动生成明细 |
| `aminer-poisoning-relation-pilot.md` | AMiner P–A、P–R、联合 poisoning 关系范围门控与阶段 F1 结论 |
| `dblp-ablation-results.md` | DBLP 四个统一 `w/o` 变体的正式组件消融与阶段 F2 审计 |
| `dvcl-topology-version-pilot.md` | DBLP 拓扑实现版本对照、阶段 F2.5 审计与 `graph_hard` 冻结结论 |
| `dvcl-hyperparameter-sensitivity.md` | DBLP 最终 DVCL 的阶段 F3 单因素敏感性与稳定性审计 |
| `model-efficiency-results.md` | 三数据集统一 11 模型的参数量、训练时间、推理延迟、峰值显存与查询成本 |
| `robust-baseline-results.md` | RoHe、HeteroGuard、FastRoHGCN |
| `openhgnn-baseline-results.md` | HGT、MAGNN、HeCo、SimpleHGN |
| `rnd-attack-results.md` | RND poisoning |
| `attack-factorial-results.md` | 攻击机制因子实验 |
| `paper-experiment-tables.md` | 旧版自动生成跨数据集表，仅用于追溯 |

## 3. 协议与审计

| 文档 | 内容 |
|---|---|
| `threat-model-audit.md` | Poisoning、迁移逃逸和模型自适应逃逸的协议边界 |
| `attack-effectiveness-audit.md` | 攻击预算、关系变化和效果审计 |
| `golden.md` | HSeCo golden 对照来源、范围和可审计性 |
| `baselines.md` | 基线实现、后端和超参数来源 |
| `hseco.md` | HSeCo 原生实现与 legacy checkpoint 语义 |
| `environment.md` | Python、Torch、DGL 与 GPU 环境 |

## 4. 工程历史

`development-roadmap.md`、`experiment-expansion.md` 和 `experiment-run-log.md` 记录阶段性规划与执行过程，不作为当前实验完成度、后续顺序或论文结论依据。

## 5. 口径规则

- 指标仅使用 Micro-F1，论文主数字以 `final-experiment-results.md` 为准。
- Poisoning、固定迁移逃逸和模型自适应逃逸分别成表，不计算跨威胁模型总平均。
- 自动生成表优先于手工历史记录；生成入口为 `python scripts/analyze_paper_results.py`。
- `final-experiment-results.html` 是本地导出副本，不是结果源；若与 Markdown 不一致，以 `final-experiment-results.md` 为准并重新导出。
- “预算验证通过”只说明 artifact 结构正确，不代表攻击足够强；强度结论需同时查看替代模型下降、目标 ASR 和实际预算利用率。
- 阶段 D 的 75 个 manifest、阶段 E 的 30 个 manifest 和阶段 F1 的 12 个 manifest 来自 dirty worktree，仅用于单种子机制筛选；论文主统计继续使用冻结的 `concat` 协议。
