# AMiner 中毒攻击关系范围 Pilot

## 实验设置

- 攻击：PRBCD、HetePRBCD；全局扰动率 $r=15\%$，攻击种子 $s_a=1$。
- 关系范围：P–A、P–R、P–A+P–R；三种范围使用相同全局扰动预算。
- 下游模型：HeteroSAGE、DVCL；训练种子 $s_t=1$；指标仅报告 Micro-F1。
- Clean Micro-F1：HeteroSAGE 84.77，DVCL 87.86。
- 扩展门槛：最佳公共关系范围的四条件平均下降至少 2 pp，且两种代理模型的下降均非负。

## 实验结果

| 范围 | 攻击 | HeteroSAGE attacked / drop | DVCL attacked / drop | Surrogate drop |
|---|---|---:|---:|---:|
| P–A | `prbcd` | 84.96 / -0.20 pp | 87.81 / 0.04 pp | 0.37 pp |
| P–A | `heteprbcd` | 84.96 / -0.20 pp | 87.57 / 0.28 pp | 2.96 pp |
| P–R | `prbcd` | 85.29 / -0.52 pp | 87.75 / 0.11 pp | 0.63 pp |
| P–R | `heteprbcd` | 84.33 / 0.44 pp | 87.60 / 0.26 pp | 0.50 pp |
| P–A+P–R | `prbcd` | 85.01 / -0.24 pp | 87.81 / 0.04 pp | 0.35 pp |
| P–A+P–R | `heteprbcd` | 85.35 / -0.59 pp | 87.70 / 0.15 pp | 0.70 pp |

| 范围 | 四条件平均下降 | 最小下降 | PRBCD surrogate | HetePRBCD surrogate |
|---|---:|---:|---:|---:|
| P–A | -0.02 pp | -0.20 pp | 0.37 pp | 2.96 pp |
| P–R | 0.07 pp | -0.52 pp | 0.63 pp | 0.50 pp |
| P–A+P–R | -0.16 pp | -0.59 pp | 0.35 pp | 0.70 pp |

## 结论

- 最佳公共关系范围：P–R，平均下降 0.07 pp。
- F1 扩展门槛：**未通过**。
- P–A HetePRBCD 在代理模型上下降 2.96 pp，但在 HeteroSAGE/DVCL 上分别为 -0.20 pp/0.28 pp，说明主要问题是跨模型迁移弱，而不是扰动预算不足。
- 不生成 attack seed 2–3 的正式全矩阵；既有 AMiner poisoning 结果只作描述性结果，不作为强鲁棒性证据。
- 完整性审计：12/12，问题数 0。
- 12 个 Pilot manifest 来自 dirty worktree；若门槛通过，正式矩阵须在冻结提交后运行。
