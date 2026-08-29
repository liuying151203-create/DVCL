# DVCL `reliability_gate` Pilot 结果

## 1. 实验设置

- 数据集：ACM、DBLP、AMiner；单配对种子 $(s_a,s_t)=(1,1)$。
- 融合：$[\alpha_i h_i^t\,\|\,(1-\alpha_i)h_i^f]$；门控只读取双视图熵、置信边界、JS 分歧、余弦一致性和范数比，不读取测试标签或 clean/attacked 成对信息。
- 损失：$L=L_c+\lambda_hL_h+\lambda_dL_d+\beta L_{aux}+\lambda_rL_r$，其中 $\lambda_h=\lambda_d=\lambda_r=1$，$\beta=0.5$，$\tau_r=1$，$d_g=16$。
- 攻击：HG Baseline 迁移攻击和针对候选完整模型重新优化的 64+64 候选自适应查询攻击；$\Delta=\{1,3,5\}$。
- 指标：仅报告 Micro-F1；本阶段是单种子机制筛选，不作显著性结论。

## 2. Clean Micro-F1

| Dataset | `feat` | `concat` | `reliability_gate` | 相对 `concat` | $\alpha$ mean±std |
|---|---:|---:|---:|---:|---:|
| ACM | 85.59 | 89.17 | 88.71 | -0.46 pp | 0.5848±0.0195 |
| DBLP | 79.98 | 89.41 | 90.05 | 0.65 pp | 0.5485±0.0212 |
| AMINER | 86.68 | 87.86 | 87.92 | 0.07 pp | 0.5874±0.0126 |

## 3. HG Baseline 目标逃逸

### ACM

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 82.00 | 82.00 | 82.00 | 82.00 | 0.00 pp | 0.00 |
| `concat` | 89.40 | 88.80 | 89.20 | 89.20 | 0.20 pp | 0.89 |
| `reliability_gate` | 88.60 | 88.60 | 88.60 | 88.60 | 0.00 pp | 0.45 |

### DBLP

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 76.88 | 76.88 | 76.88 | 76.88 | 0.00 pp | 0.00 |
| `concat` | 89.25 | 83.60 | 74.19 | 71.51 | 17.74 pp | 21.08 |
| `reliability_gate` | 90.05 | 84.41 | 75.81 | 70.43 | 19.62 pp | 22.99 |

### AMINER

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 87.77 | 87.77 | 87.77 | 87.77 | 0.00 pp | 0.00 |
| `concat` | 88.49 | 88.49 | 88.49 | 88.49 | 0.00 pp | 0.00 |
| `reliability_gate` | 89.93 | 89.69 | 89.93 | 89.93 | 0.00 pp | 0.27 |


## 4. 模型自适应目标逃逸

### ACM

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 72.00 | 72.00 | 72.00 | 72.00 | 0.00 pp | 0.00 |
| `concat` | 86.00 | 86.00 | 86.00 | 86.00 | 0.00 pp | 0.00 |
| `reliability_gate` | 84.00 | 84.00 | 84.00 | 84.00 | 0.00 pp | 0.00 |

### DBLP

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 76.00 | 76.00 | 76.00 | 76.00 | 0.00 pp | 0.00 |
| `concat` | 88.00 | 54.00 | 44.00 | 42.00 | 46.00 pp | 52.27 |
| `reliability_gate` | 88.00 | 58.00 | 44.00 | 40.00 | 48.00 pp | 54.55 |

### AMINER

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 64.00 | 64.00 | 64.00 | 64.00 | 0.00 pp | 0.00 |
| `concat` | 66.00 | 66.00 | 66.00 | 66.00 | 0.00 pp | 0.00 |
| `reliability_gate` | 70.00 | 70.00 | 70.00 | 70.00 | 0.00 pp | 0.00 |

## 5. 门控行为与验收

| Dataset | Attack | $\alpha$ clean→attack | std clean→attack | View disagreement clean→attack |
|---|---|---:|---:|---:|
| ACM | `hg_baseline` | 0.5850→0.5851 | 0.0186→0.0184 | 11.40→11.60 |
| ACM | `adaptive_query` | 0.5852→0.5835 | 0.0212→0.0213 | 10.00→10.00 |
| DBLP | `hg_baseline` | 0.5485→0.5457 | 0.0220→0.0214 | 18.01→27.96 |
| DBLP | `adaptive_query` | 0.5523→0.5475 | 0.0216→0.0247 | 20.00→42.00 |
| AMINER | `hg_baseline` | 0.5880→0.5879 | 0.0118→0.0118 | 6.00→6.47 |
| AMINER | `adaptive_query` | 0.5817→0.5797 | 0.0177→0.0167 | 16.00→16.00 |

- DBLP 自适应 $\Delta=5$ 相对 `concat` 增益：-2.00 pp（门槛 $\geq5$ pp）。
- 三数据集最大 clean 损失：0.46 pp（门槛 $\leq1.5$ pp）。
- ACM/AMiner 最大攻击后损失：2.00 pp（门槛 $\leq2$ pp）。
- 门控非塌缩：否（clean $\alpha$ std $\geq0.02$，极端路由比例均 $<95\%$）。
- 阶段 E Pilot 判定：**未通过**；下一步保持当前结果不变，进入带训练时结构扰动的 `reliability_gate_aug` Pilot。
- 完整性审计：clean 3/3，攻击评估 12/12，逻辑结果 18/18，问题数 0。
- 15 个候选运行 manifest 标记为 dirty worktree；本 Pilot 仅用于机制筛选，通过后仍须在冻结提交上重跑正式统计。
