# 统一协议基线

## 已接入模型

首批原生基线为 HAN 和 HeteroSAGE。两者与 HSeCo、DVCL 共用 clean、split、
attack artifact、训练种子、训练轮数、早停规则、checkpoint 和指标实现。

| 模型 | 输入图 | 非目标节点特征 | 主要配置 |
|---|---|---|---|
| HAN | 未净化的元路径可达图 | 不适用 | \(d=64\), \(K=8\), \(p=0.3\), \(\eta=0.005\) |
| HeteroSAGE | 原始异构关系图 | 与旧基线一致采用零向量 | \(L=2\), \(d=64\), \(p=0.5\), \(\eta=0.01\) |

当前实现是统一协议下的独立原生实现，不能直接替代其他论文在不同数据划分、
攻击文件或训练协议下报告的数字。

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

下表先在每个训练种子内对攻击条件取平均，再计算 5 个训练种子间的均值和样本
标准差，单位为百分数。Attack Average 包含全部 10 个攻击条件，不包含 clean。

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

1. ACM 上 HAN 与 HeteroSAGE 的 Attack Average 分别为 87.95 和 87.55，DVCL
   为 88.16，DVCL 在整体攻击平均上保持最高结果。
2. DBLP PRBCD 下 HAN 达到 90.99，但在 HetePRBCD 下下降到 27.26，说明该模型
   对不同异构攻击族的表现差异显著，需结合关系级边变化进一步诊断。
3. HeteroSAGE 在 DBLP HetePRBCD 下优于 HAN，但 56.51 仍明显低于 HSeCo 的
   76.30 和 DVCL 的 82.85。
4. DBLP Attack Average 上，DVCL 达到 85.88，分别比 HAN 和 HeteroSAGE 高
   26.76 和 20.13 个百分点。

## 后续鲁棒基线

RoHe、HeteroGuard 和 FastRo-HGCN 需要分别核对官方代码、依赖、输入特征和
攻击协议后再接入。无法证明源码和协议一致时，只能标记为独立复现或外部后端，
不能标记为官方等效实现。
