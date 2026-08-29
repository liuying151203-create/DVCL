# DVCL 阶段 E 防御改进结果

## 1. 实验设置

- 数据集：ACM、DBLP、AMiner；单配对种子 $(s_a,s_t)=(1,1)$。
- 对照：阶段 D 的 `feat`、`concat` 和已有门控；候选为 `reliability_gate` 与 `reliability_gate_aug`。
- `reliability_gate` 使用双视图熵、置信边界、JS 分歧、余弦一致性和范数比预测节点级拓扑权重 $\alpha_i$，不读取测试标签或 clean/attacked 成对信息。
- `reliability_gate_aug` 仅增加训练时 10% 拓扑边随机重连与增强损失 $L_{aug}$，推理结构不变。
- 攻击：HG Baseline 迁移逃逸和针对每个候选完整模型重新生成的 64+64 候选自适应查询攻击，$\Delta=\{1,3,5\}$。
- 指标：仅报告 Micro-F1；本阶段为机制 Pilot，不作显著性结论。

## 2. Clean Micro-F1

| Model | ACM | DBLP | AMiner | 最大 `concat` 损失 |
|---|---:|---:|---:|---:|
| `feat` | 85.59 | 79.98 | 86.68 | 9.43 pp |
| `concat` | 89.17 | 89.41 | 87.86 | — |
| `reliability_gate` | 88.71 | 90.05 | 87.92 | 0.46 pp |
| `reliability_gate_aug` | 88.46 | 88.82 | 87.83 | 0.71 pp |

## 3. 自适应目标逃逸

| Dataset | Model | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 | ASR@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| ACM | `concat` | 86.00 | 86.00 | 86.00 | 86.00 | 0.00 pp | 0.00 |
| ACM | `reliability_gate` | 84.00 | 84.00 | 84.00 | 84.00 | 0.00 pp | 0.00 |
| ACM | `reliability_gate_aug` | 84.00 | 84.00 | 82.00 | 82.00 | 2.00 pp | 2.38 |
| DBLP | `concat` | 88.00 | 54.00 | 44.00 | 42.00 | 46.00 pp | 52.27 |
| DBLP | `reliability_gate` | 88.00 | 58.00 | 44.00 | 40.00 | 48.00 pp | 54.55 |
| DBLP | `reliability_gate_aug` | 80.00 | 64.00 | 48.00 | 44.00 | 36.00 pp | 45.00 |
| AMiner | `concat` | 66.00 | 66.00 | 66.00 | 66.00 | 0.00 pp | 0.00 |
| AMiner | `reliability_gate` | 70.00 | 70.00 | 70.00 | 70.00 | 0.00 pp | 0.00 |
| AMiner | `reliability_gate_aug` | 72.00 | 72.00 | 72.00 | 72.00 | 0.00 pp | 0.00 |

## 4. HG Baseline $\Delta=5$

| Dataset | Model | Clean target | Attacked | Drop | ASR |
|---|---|---:|---:|---:|---:|
| ACM | `concat` | 89.40 | 89.20 | 0.20 pp | 0.89 |
| ACM | `reliability_gate` | 88.60 | 88.60 | 0.00 pp | 0.45 |
| ACM | `reliability_gate_aug` | 87.80 | 88.20 | -0.40 pp | 0.46 |
| DBLP | `concat` | 89.25 | 71.51 | 17.74 pp | 21.08 |
| DBLP | `reliability_gate` | 90.05 | 70.43 | 19.62 pp | 22.99 |
| DBLP | `reliability_gate_aug` | 87.10 | 70.97 | 16.13 pp | 19.75 |
| AMiner | `concat` | 88.49 | 88.49 | 0.00 pp | 0.00 |
| AMiner | `reliability_gate` | 89.93 | 89.93 | 0.00 pp | 0.27 |
| AMiner | `reliability_gate_aug` | 90.41 | 90.41 | 0.00 pp | 0.00 |

## 5. 分析与结论

- 未增强门控在 DBLP 自适应 $\Delta=5$ 下仅将 $\alpha$ 从 0.5523 降到 0.5475，攻击后 Micro-F1 为 40，较 `concat` 低 2 pp。
- 结构增强使 DBLP 的 $\alpha$ 从 0.5240 降到 0.5095，并将攻击后 Micro-F1 提高到 44、ASR 降到 45%；但增益只有 2 pp，未达到预注册的 5 pp 门槛。
- 增强版在 ACM 自适应攻击下由 `concat` 的 86 降至 82，损失 4 pp，超过 2 pp 安全门槛；两个候选也均未满足三数据集门控非塌缩门槛。
- 两个候选的 clean 损失都在 1.5 pp 以内，说明失败原因不是 clean 性能崩溃，而是可靠性信号不足以稳定识别受攻击拓扑视图。
- 阶段 E 判定为未通过。停止继续基于单种子结果调参，不扩展候选到多种子；后续保留 `concat` 作为论文主模型，并把 DBLP 自适应脆弱性作为明确局限。
- `reliability_gate` 与 `reliability_gate_aug` 均完成 3/3 clean、12/12 物理攻击评估和 18/18 逻辑结果，审计问题均为 0；两组各 15 个 manifest 来自 dirty worktree，只作为机制筛选证据。

详细结果分别见 `dvcl-reliability-gate-results.md` 和 `dvcl-reliability-gate-augmentation-results.md`；机器可读审计位于 `outputs/analysis/dvcl_reliability_gate_pilot_v1/` 与 `outputs/analysis/dvcl_reliability_gate_aug_pilot_v1/`。
