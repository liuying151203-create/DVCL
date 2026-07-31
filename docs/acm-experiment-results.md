# ACM 实验结果与论文表格草案

## 1. 文档定位

本文档汇总当前已完成的 ACM GPU 实验，并按接近论文主表、攻击平均表和
消融表的格式整理结果。当前结果可以作为 ACM 部分的论文表格草案，但不能
替代尚未完成的 DBLP 实验和其他基线对比。

结果生成日期为 2026-07-30。所有 `dvcl_main` 运行均来自 Git commit
`63fcc82cbf4bd8f54690a334d00db81f532cf7eb`，且 manifest 记录的工作树状态
均为 clean。

## 2. 实验协议

- 数据集：ACM；
- split：`paper_seed_1`；
- attack seed：1；
- train seed：1、2、3、4、5；
- 主实验攻击：clean、PRBCD 5%–25%、HetePRBCD 5%–25%；
- 消融攻击：clean、PRBCD 5%/15%/25%、HetePRBCD 5%/15%/25%；
- 训练预算：最多 200 epochs，patience 100；
- 设备：Tesla V100-PCIE-32GB；
- Torch：2.1.2+cu121；
- DGL：1.1.3+cu121；
- PyG：2.5.3。

表中数值均为 5 个 train seed 的 `均值 ± 样本标准差`，并统一换算为百分数。
由于当前任务是单标签多分类，Accuracy 与 Micro-F1 完全相同，后续论文结果表
统一使用 `Accuracy / Micro-F1` 表示。

## 3. 完整性审计

| 实验部分 | 预期运行 | 已完成 | 失败 | 状态 |
|---|---:|---:|---:|---|
| HSeCo paper reproduction，ACM clean | 5 | 5 | 0 | 完成 |
| ACM 主实验，HSeCo 与 DVCL | 110 | 110 | 0 | 完成 |
| ACM DVCL 组件消融 | 140 | 140 | 0 | 完成 |
| ACM `dvcl_main` 合计 | 250 | 250 | 0 | 完成 |
| DBLP 主实验 | 110 | 0 | 0 | 尚未开始 |

对 250 个 `dvcl_main` 运行的自动审计结果：

- 50 个实验条件均包含 train seed 1–5；
- 所有 `status.json` 均为 `completed`；
- 每次运行均包含 checkpoint、history、metrics、status 和 manifest；
- 所有 manifest schema 均为版本 2；
- 所有运行均记录 `cuda:0` 和 Tesla V100；
- 同一实验条件下的 clean、split 和 attack artifact 哈希一致；
- 所有运行使用同一个 Git commit，且 `git_dirty=false`；
- DVCL `default` 与消融 `full` 在重叠的 7 个条件上结果完全一致。

## 4. ACM 主实验结果

**表 1：ACM 上 HSeCo 与 DVCL 的鲁棒节点分类结果。** 加粗表示表现更好的
模型；`Δ` 表示 DVCL 相对 HSeCo 的绝对百分点提升。

| Attack | Rate | HSeCo Accuracy / Micro-F1 | DVCL Accuracy / Micro-F1 | Δ |
|---|---:|---:|---:|---:|
| Clean | — | 87.63 ± 0.49 | **88.63 ± 0.66** | +1.01 |
| PRBCD | 5% | 85.84 ± 0.48 | **87.94 ± 0.47** | +2.10 |
| PRBCD | 10% | 86.23 ± 0.42 | **87.82 ± 0.30** | +1.59 |
| PRBCD | 15% | 85.52 ± 0.82 | **88.00 ± 0.35** | +2.48 |
| PRBCD | 20% | 86.09 ± 0.31 | **87.80 ± 0.43** | +1.70 |
| PRBCD | 25% | 86.14 ± 0.28 | **88.16 ± 0.33** | +2.02 |
| HetePRBCD | 5% | 87.41 ± 0.34 | **88.48 ± 0.71** | +1.07 |
| HetePRBCD | 10% | 86.69 ± 0.97 | **88.56 ± 0.53** | +1.87 |
| HetePRBCD | 15% | 86.13 ± 1.34 | **88.43 ± 0.79** | +2.30 |
| HetePRBCD | 20% | 86.25 ± 1.20 | **88.16 ± 0.55** | +1.90 |
| HetePRBCD | 25% | 86.71 ± 0.86 | **88.29 ± 0.58** | +1.58 |

## 5. 攻击族平均结果

攻击族平均采用“两级平均”：先对每个 train seed 的攻击条件取平均，再在
5 个 train seed 之间计算均值和样本标准差。Attack Average 包含全部 10 个
PRBCD/HetePRBCD 条件，不包含 clean。

**表 2：ACM 攻击族平均结果。**

| Condition | HSeCo Accuracy / Micro-F1 | DVCL Accuracy / Micro-F1 | Δ |
|---|---:|---:|---:|
| Clean | 87.63 ± 0.49 | **88.63 ± 0.66** | +1.01 |
| PRBCD Average | 85.96 ± 0.27 | **87.94 ± 0.26** | +1.98 |
| HetePRBCD Average | 86.64 ± 0.82 | **88.38 ± 0.59** | +1.75 |
| Attack Average | 86.30 ± 0.48 | **88.16 ± 0.38** | +1.86 |

## 6. DVCL 组件消融

消融配置含以下四个变体：

- Full DVCL：拓扑视图、特征视图和跨视图对比学习全部启用；
- w/o Cross-view CL：保留双视图，移除跨视图对比损失；
- Topology only：仅使用语义拓扑视图；
- Feature only：仅使用特征 KNN 视图。

消融中的 PRBCD/HetePRBCD Average 分别包含 5%、15% 和 25% 三个攻击率；
Attack Average 包含两个攻击族的全部 6 个条件。

**表 3：DVCL 组件消融的 Accuracy / Micro-F1。**

| Variant | Clean | PRBCD Average | HetePRBCD Average | Attack Average |
|---|---:|---:|---:|---:|
| Full DVCL | **88.63 ± 0.66** | **88.04 ± 0.31** | **88.40 ± 0.63** | **88.22 ± 0.42** |
| w/o Cross-view CL | 88.12 ± 0.53 | 87.08 ± 0.78 | 87.37 ± 0.91 | 87.22 ± 0.68 |
| Topology only | 87.59 ± 0.50 | 85.66 ± 0.49 | 86.63 ± 0.76 | 86.14 ± 0.51 |
| Feature only | 86.47 ± 0.76 | 86.47 ± 0.76 | 86.47 ± 0.76 | 86.47 ± 0.76 |

## 7. HSeCo 复现审计

该表用于协议审计，不应与主表中的统一 checkpoint 语义混用。

**表 4：HSeCo clean 复现协议与统一主协议对照。**

| Protocol | Checkpoint semantics | Accuracy / Micro-F1 | Runs |
|---|---|---:|---:|
| `hseco_paper_reproduction` | Legacy node-model checkpoint | 87.63 ± 0.46 | 5 |
| `dvcl_main` HSeCo clean | Complete native checkpoint | 87.63 ± 0.49 | 5 |

两组结果非常接近，但由于 checkpoint 语义不同，不能把其中一组直接替代另一组。
旧 HSeCo CPU golden 的严格零差异结论仅适用于相同 CPU、`seed_1` split 和
train seed 1 的受控迁移审计。

## 8. 结果解读

1. DVCL 在 ACM clean 和全部 10 个攻击条件下均优于 HSeCo。
2. 在全部攻击条件上，DVCL 相对 HSeCo 的 Accuracy / Micro-F1 平均提升
   1.86 个百分点。
3. 移除跨视图对比学习后，Attack Average 下降 0.99 个百分点。
4. 仅保留拓扑视图时，Attack Average 相对完整 DVCL 下降 2.07 个百分点，
   说明单一拓扑视图对结构攻击更加敏感。
5. Feature only 在 clean 和所有攻击条件上结果完全一致，这是因为当前
   PRBCD/HetePRBCD artifact 只改变拓扑，而该变体只读取固定特征 KNN 视图。
6. 攻击率与性能不严格单调。攻击率表示扰动预算，不保证每个具体 artifact
   对当前模型形成严格单调的难度顺序；论文中应避免把非单调波动解释为异常提升。

## 9. 发表前剩余工作

- 完成 DBLP clean、split 和 10 个攻击 artifact；
- 完成 DBLP 110 次主实验并生成与表 1 相同格式的结果；
- 扩大 HSeCo 攻击场景和 DVCL 的 golden 覆盖；
- 根据论文定位补充其他异构图鲁棒基线；
- 对主要结论增加配对显著性检验和效应量；
- 将原始 outputs、artifact 哈希、环境报告和 Git commit 独立归档；
- 最终排版时将 Markdown 表转换为 LaTeX，并统一有效数字和加粗规则。

## 10. 数据来源

- 单次运行：`outputs/summaries/dvcl_main/runs.csv`；
- 条件均值与标准差：`outputs/summaries/dvcl_main/summary.csv`；
- Attack Average：`outputs/summaries/dvcl_main/attack_average.csv`；
- HSeCo paper reproduction：
  `outputs/summaries/hseco_paper_reproduction/summary.csv`；
- 原始运行：`outputs/runs/dvcl_main/`。
