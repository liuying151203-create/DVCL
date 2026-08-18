# 统一协议基线

> 协议审计状态：HAN/HeteroSAGE 及 RoHe/HeteroGuard/FastRoHGCN 的正式矩阵已经
> 完成；OpenHGNN 基线正式矩阵待运行。

## 已接入模型

首批原生基线为 HAN 和 HeteroSAGE。两者与 HSeCo、DVCL 共用 clean、split、
attack artifact、训练种子、训练轮数、早停规则、checkpoint 和指标实现。

| 模型 | 输入图 | 非目标节点特征 | 主要配置 |
|---|---|---|---|
| HAN | 未净化的元路径可达图 | 不适用 | \(d=64\), \(K=8\), \(p=0.3\), \(\eta=0.005\) |
| HeteroSAGE | 原始异构关系图 | 与旧基线一致采用零向量 | \(L=2\), \(d=64\), \(p=0.5\), \(\eta=0.01\) |

当前实现是统一协议下的独立原生实现，不能直接替代其他论文在不同数据划分、
攻击文件或训练协议下报告的数字。

### OpenHGNN 通用基线

HGT、MAGNN、HeCo 和 SimpleHGN 使用固定 OpenHGNN 0.4.1 修订 `27a483e`。训练仍由
本项目读取冻结 artifact、执行统一 split/attack 校验、早停、checkpoint 和指标汇总。

| 模型 | 主要配置 | 协议说明 |
|---|---|---|
| HGT | $d=64$, $L=2$, $K=8$, $p=0.4$, $\eta=10^{-3}$ | 官方全图 HGT 编码器与可学习类型特征 |
| MAGNN | $d=64$, $L=4$, $K=8$, $d_a=128$, $M=5$, $\eta=0.005$ | 每条元路径每节点采样 $M$ 个实例；修正 0.4.1 两处执行缺陷 |
| HeCo | $d=64$, $p_f=0.3$, $p_a=0.5$, $\tau=0.8$, $\lambda=0.5$ | 对比预训练后训练单个 seeded 线性分类器 |
| SimpleHGN | $d=256$, $L=3$, $K=8$, $d_e=64$, $\beta=0.05$ | 官方 relation-aware attention 编码器 |

运行时会校验四个官方模型文件的 SHA-256。上述配置是统一协议的冻结复现配置，不等同
于在其他数据版本或划分上报告的 OpenHGNN 默认结果。

## 实验矩阵

`configs/protocols/baseline_main.yaml` 包含：

- 数据集：ACM、DBLP；
- 模型：HAN、HeteroSAGE；
- 条件：clean、PRBCD 5%–25%、HetePRBCD 5%–25%；
- 训练种子：1–5；
- 总计：220 次运行。

```bash
source scripts/activate_gpu_env.sh
python scripts/run_suite.py --config configs/protocols/baseline_main.yaml --dry-run
python scripts/run_suite.py --config configs/protocols/baseline_main.yaml --continue-on-error
python scripts/summarize_results.py \
  --run-root outputs/runs/baseline_main \
  --output-dir outputs/summaries/baseline_main
```

HAN 和 HeteroSAGE 已通过真实 ACM clean artifact 的单 epoch CPU 与 Tesla V100
GPU Runner pilot。正式矩阵 220/220 次运行均为 `completed`，44 个条件均包含
5 个训练种子，全部攻击验证通过，checkpoint、history、metrics、status 和
manifest 完整。

## 实验结果

结果均为 5 个训练种子的均值 ± 样本标准差，单位为百分数，统一记为
`Accuracy / Micro-F1`。

### ACM

| Attack | Rate | HAN | HeteroSAGE |
|---|---:|---:|---:|
| Clean | — | **91.16 ± 0.28** | 88.95 ± 2.64 |
| PRBCD | 5% | **90.15 ± 1.42** | 88.41 ± 1.55 |
| PRBCD | 10% | **88.23 ± 1.14** | 88.17 ± 1.49 |
| PRBCD | 15% | 87.43 ± 0.41 | **87.96 ± 1.09** |
| PRBCD | 20% | **86.82 ± 0.66** | 86.60 ± 2.21 |
| PRBCD | 25% | 86.59 ± 1.14 | **86.60 ± 2.36** |
| HetePRBCD | 5% | **90.20 ± 0.94** | 87.72 ± 1.60 |
| HetePRBCD | 10% | **89.74 ± 1.06** | 88.15 ± 2.11 |
| HetePRBCD | 15% | 87.38 ± 1.80 | **87.58 ± 1.41** |
| HetePRBCD | 20% | 87.03 ± 2.50 | **87.52 ± 1.64** |
| HetePRBCD | 25% | 85.89 ± 2.97 | **86.76 ± 1.10** |

### DBLP

| Attack | Rate | HAN | HeteroSAGE |
|---|---:|---:|---:|
| Clean | — | **92.84 ± 0.31** | 80.49 ± 0.38 |
| PRBCD | 5% | **90.88 ± 0.87** | 75.93 ± 0.55 |
| PRBCD | 10% | **90.94 ± 0.78** | 74.49 ± 1.28 |
| PRBCD | 15% | **91.78 ± 0.34** | 74.81 ± 1.32 |
| PRBCD | 20% | **90.96 ± 0.79** | 75.51 ± 1.26 |
| PRBCD | 25% | **90.40 ± 0.35** | 74.19 ± 1.19 |
| HetePRBCD | 5% | 27.11 ± 0.01 | **62.54 ± 5.47** |
| HetePRBCD | 10% | 27.10 ± 0.00 | **62.79 ± 4.84** |
| HetePRBCD | 15% | 27.10 ± 0.00 | **56.37 ± 6.15** |
| HetePRBCD | 20% | 27.11 ± 0.01 | **52.33 ± 8.22** |
| HetePRBCD | 25% | 27.87 ± 0.37 | **48.49 ± 6.62** |

### 攻击平均

先在每个训练种子内对攻击条件取平均，再计算种子间的均值和样本标准差。
Attack Average 包含全部 10 个攻击条件，不包含 clean。

| Dataset | Condition | HAN | HeteroSAGE |
|---|---|---:|---:|
| ACM | Clean | **91.16 ± 0.28** | 88.95 ± 2.64 |
| ACM | PRBCD Average | **87.84 ± 0.81** | 87.55 ± 1.42 |
| ACM | HetePRBCD Average | **88.05 ± 1.69** | 87.54 ± 1.31 |
| ACM | Attack Average | **87.95 ± 1.24** | 87.55 ± 1.35 |
| DBLP | Clean | **92.84 ± 0.31** | 80.49 ± 0.38 |
| DBLP | PRBCD Average | **90.99 ± 0.19** | 74.98 ± 0.96 |
| DBLP | HetePRBCD Average | 27.26 ± 0.08 | **56.51 ± 6.14** |
| DBLP | Attack Average | 59.12 ± 0.08 | **65.75 ± 3.42** |

全部攻击率下的 HAN、HeteroSAGE、HSeCo 和 DVCL 对比见
`docs/paper-experiment-tables.md`。

**审计状态：暂定。** 220 次运行均使用 Tesla V100 和 `cuda:0`，但 manifest
记录为提交 `ec4ea1d` 且 `git_dirty=true`。结果矩阵与数值完整，正式论文引用前
仍应在包含本次实现的干净提交上强制复跑。

## 结果分析

1. 攻击链路有效。ACM 25% 条件均通过 4358/4358 的唯一边预算检查；计入反向
   关系后实际修改 8716 条有向边。HAN 第一元路径图相对 clean 的 Jaccard 在
   PRBCD/HetePRBCD 下分别降至 0.643/0.657，模型 diagnostics 与攻击图一致。
2. ACM 上 HAN 在 PRBCD 25% 和 HetePRBCD 25% 下相对 clean 分别下降 4.57 和
   5.27 个百分点；HeteroSAGE 分别下降 2.35 和 2.19 个百分点，并非没有下降。
3. 两个模型没有显式净化模块，但仍保留节点特征、自环或根节点变换。ACM 的
   固定特征视图单独即可达到 86.47，因此纯拓扑 poisoning 不必导致分类崩溃。
   当前攻击还是 `adaptive=false` 的全局迁移攻击，模型会在扰动图上重新训练。
4. ACM Attack Average 上 HAN、HeteroSAGE 和 DVCL 分别为 87.95、87.55 和
   88.16，DVCL 的平均优势较小，不能据此宣称 ACM 上存在大幅鲁棒性提升。
5. DBLP PRBCD 下 HAN 达到 90.99，但在 HetePRBCD 下下降到 27.26，说明该模型
   对不同异构攻击族的表现差异显著，需结合关系级边变化进一步诊断。
6. HeteroSAGE 在 DBLP HetePRBCD 下优于 HAN，但 56.51 仍明显低于 HSeCo 的
   76.30 和 DVCL 的 82.85。
7. DBLP Attack Average 上，DVCL 达到 85.88，分别比 HAN 和 HeteroSAGE 高
   26.76 和 20.13 个百分点。

## 后续鲁棒基线

RoHe、HeteroGuard 和 FastRo-HGCN 已接入统一原生协议，并明确标记为独立复现；在
无法证明源码、预处理和协议逐项一致前，不标记为官方等效实现。正式 poisoning、RND
和 HG Baseline target evasion 矩阵及运行顺序见 `docs/experiment-expansion.md`。

三种鲁棒基线的 ACM/DBLP poisoning 正式矩阵已完成 330/330 次，Micro-F1 结果和分析
见 `docs/robust-baseline-results.md`。

HGT、MAGNN、HeCo 和 SimpleHGN 的 ACM/DBLP poisoning suite 为
`configs/protocols/openhgnn_baselines_poisoning_v1.yaml`，共 440 次正式运行。
