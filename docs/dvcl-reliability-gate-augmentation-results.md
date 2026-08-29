# DVCL `reliability_gate_aug` Pilot 结果

## 1. 实验设置

- 数据集：ACM、DBLP、AMiner；单配对种子 $(s_a,s_t)=(1,1)$。
- 融合：$[\alpha_i h_i^t\,\|\,(1-\alpha_i)h_i^f]$；门控只读取双视图熵、置信边界、JS 分歧、余弦一致性和范数比，不读取测试标签或 clean/attacked 成对信息。
- 损失：$L=L_c+\lambda_hL_h+\lambda_dL_d+\beta L_{aux}+\lambda_rL_r$，其中 $\lambda_h=\lambda_d=\lambda_r=1$，$\beta=0.5$，$\tau_r=1$，$d_g=16$。
- 攻击：HG Baseline 迁移攻击和针对候选完整模型重新优化的 64+64 候选自适应查询攻击；$\Delta=\{1,3,5\}$。
- 训练增强：仅在训练阶段随机重连 10% 拓扑图边，并加入 $L_{aug}$；推理结构与 `reliability_gate` 相同。
- 指标：仅报告 Micro-F1；本阶段是单种子机制筛选，不作显著性结论。

## 2. Clean Micro-F1

| Dataset | `feat` | `concat` | `reliability_gate_aug` | 相对 `concat` | $\alpha$ mean±std |
|---|---:|---:|---:|---:|---:|
| ACM | 85.59 | 89.17 | 88.46 | -0.71 pp | 0.5797±0.0198 |
| DBLP | 79.98 | 89.41 | 88.82 | -0.59 pp | 0.5247±0.0256 |
| AMINER | 86.68 | 87.86 | 87.83 | -0.02 pp | 0.5669±0.0111 |

## 3. HG Baseline 目标逃逸

### ACM

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 82.00 | 82.00 | 82.00 | 82.00 | 0.00 pp | 0.00 |
| `concat` | 89.40 | 88.80 | 89.20 | 89.20 | 0.20 pp | 0.89 |
| `reliability_gate_aug` | 87.80 | 88.00 | 88.20 | 88.20 | -0.40 pp | 0.46 |

### DBLP

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 76.88 | 76.88 | 76.88 | 76.88 | 0.00 pp | 0.00 |
| `concat` | 89.25 | 83.60 | 74.19 | 71.51 | 17.74 pp | 21.08 |
| `reliability_gate_aug` | 87.10 | 81.99 | 75.27 | 70.97 | 16.13 pp | 19.75 |

### AMINER

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 87.77 | 87.77 | 87.77 | 87.77 | 0.00 pp | 0.00 |
| `concat` | 88.49 | 88.49 | 88.49 | 88.49 | 0.00 pp | 0.00 |
| `reliability_gate_aug` | 90.41 | 90.41 | 90.41 | 90.41 | 0.00 pp | 0.00 |


## 4. 模型自适应目标逃逸

### ACM

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 72.00 | 72.00 | 72.00 | 72.00 | 0.00 pp | 0.00 |
| `concat` | 86.00 | 86.00 | 86.00 | 86.00 | 0.00 pp | 0.00 |
| `reliability_gate_aug` | 84.00 | 84.00 | 82.00 | 82.00 | 2.00 pp | 2.38 |

### DBLP

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 76.00 | 76.00 | 76.00 | 76.00 | 0.00 pp | 0.00 |
| `concat` | 88.00 | 54.00 | 44.00 | 42.00 | 46.00 pp | 52.27 |
| `reliability_gate_aug` | 80.00 | 64.00 | 48.00 | 44.00 | 36.00 pp | 45.00 |

### AMINER

| Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---:|---:|---:|---:|---:|---:|
| `feat` | 64.00 | 64.00 | 64.00 | 64.00 | 0.00 pp | 0.00 |
| `concat` | 66.00 | 66.00 | 66.00 | 66.00 | 0.00 pp | 0.00 |
| `reliability_gate_aug` | 72.00 | 72.00 | 72.00 | 72.00 | 0.00 pp | 0.00 |

## 5. 门控行为与验收

| Dataset | Attack | $\alpha$ clean→attack | std clean→attack | View disagreement clean→attack |
|---|---|---:|---:|---:|
| ACM | `hg_baseline` | 0.5801→0.5801 | 0.0192→0.0187 | 12.60→12.80 |
| ACM | `adaptive_query` | 0.5822→0.5808 | 0.0226→0.0231 | 8.00→14.00 |
| DBLP | `hg_baseline` | 0.5241→0.5128 | 0.0288→0.0308 | 21.24→31.99 |
| DBLP | `adaptive_query` | 0.5240→0.5095 | 0.0311→0.0342 | 22.00→50.00 |
| AMINER | `hg_baseline` | 0.5675→0.5674 | 0.0111→0.0111 | 5.52→5.76 |
| AMINER | `adaptive_query` | 0.5611→0.5597 | 0.0167→0.0159 | 12.00→12.00 |

- DBLP 自适应 $\Delta=5$ 相对 `concat` 增益：2.00 pp（门槛 $\geq5$ pp）。
- 三数据集最大 clean 损失：0.71 pp（门槛 $\leq1.5$ pp）。
- ACM/AMiner 最大攻击后损失：4.00 pp（门槛 $\leq2$ pp）。
- 门控非塌缩：否（clean $\alpha$ std $\geq0.02$，极端路由比例均 $<95\%$）。
- 阶段 E Pilot 判定：**未通过**；下一步停止继续追逐单种子测试结果，保留 `concat` 作为主模型并结束阶段 E。
- 完整性审计：clean 3/3，攻击评估 12/12，逻辑结果 18/18，问题数 0。
- 15 个候选运行 manifest 标记为 dirty worktree；本 Pilot 仅用于机制筛选，通过后仍须在冻结提交上重跑正式统计。
