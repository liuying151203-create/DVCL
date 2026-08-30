# DBLP 组件消融结果

## 实验设置

- 数据集：DBLP；划分与攻击种子 $(s_s,s_a)=(1,1)$，训练种子 $s_t=1,\ldots,5$。
- 攻击：PRBCD、HetePRBCD，$r\in\{5\%,15\%,25\%\}$；均为全局 poisoning。
- 变体：Full DVCL、w/o Cross-view CL、w/o Feature View、w/o Topology View。
- 训练：$E_{max}=200$，$P=100$；仅报告 Micro-F1。
- 攻击平均先在每个训练种子内跨扰动率平均，再计算五种子的均值与标准差。

## 实验结果

| Variant | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |
|---|---:|---:|---:|---:|
| Full DVCL | 89.55 ± 0.64 | 87.09 ± 0.77 | 82.04 ± 0.76 | 84.56 ± 0.74 |
| w/o Cross-view CL | 88.02 ± 1.53 | 86.20 ± 0.62 | 80.07 ± 1.16 | 83.14 ± 0.76 |
| w/o Feature View | 89.26 ± 0.58 | 86.88 ± 1.18 | 71.16 ± 1.92 | 79.02 ± 1.54 |
| w/o Topology View | 79.79 ± 0.47 | 79.79 ± 0.47 | 79.79 ± 0.47 | 79.79 ± 0.47 |

## 配对贡献

正值表示 Full DVCL 优于对应消融；统计单元为五个训练种子。

| Ablation | Clean | PRBCD Avg. | HetePRBCD Avg. | Attack Avg. |
|---|---:|---:|---:|---:|
| w/o Cross-view CL | 1.53 ± 1.36 | 0.88 ± 0.38 | 1.97 ± 0.71 | 1.42 ± 0.35 |
| w/o Feature View | 0.30 ± 0.79 | 0.21 ± 0.56 | 10.88 ± 1.19 | 5.54 ± 0.83 |
| w/o Topology View | 9.76 ± 0.93 | 7.30 ± 1.03 | 2.25 ± 0.86 | 4.77 ± 0.93 |

## 分析

- Full DVCL 的 Attack Average 为 84.56。
- 移除跨视图对比学习后平均下降 1.42 pp；移除特征视图后下降 5.54 pp。
- 移除拓扑视图后的配对差异为 4.77 pp；该变体在纯结构攻击下逐种子保持不变，符合特征视图不读取攻击图的实现语义。
- Full DVCL 与 DBLP 主实验逐条件逐种子一致：35/35。
- 完整性审计：140/140，问题数 0；manifest commit 为 `4434ecf77fb19f3d8039e9732ba1593eafb5828e`。
