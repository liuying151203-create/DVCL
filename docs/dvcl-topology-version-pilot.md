# DVCL 拓扑实现版本 Pilot

## 实验设置

- 数据集：DBLP；配对重复 $(s_a,s_t)=(1,1),(2,2),(3,3)$。
- 变体：当前 `graph_hard` 与取消第二级硬阈值的 `graph_no_filter`。
- 条件：clean、HetePRBCD $r=25\%$ poisoning、每模型独立优化的 64+64 候选自适应目标逃逸。
- 预算：$\Delta=\{1,3,5\}$；$E_{max}=200$，$P=100$；仅报告 Micro-F1。
- 两个变体均令 $\lambda_h=1$，其余模型、训练和攻击超参数保持一致。

## Poisoning 与 Clean

| Variant | Clean | HetePRBCD 25% | Drop |
|---|---:|---:|---:|
| Graph + hard filter + $L_{HAN}$ | 89.13 ± 0.31 | 83.75 ± 2.37 | 5.38 pp |
| Graph + no second filter + $L_{HAN}$ | 88.60 ± 0.26 | 82.08 ± 1.19 | 6.53 pp |

## 自适应目标逃逸

| Variant | Clean target | $\Delta=1$ | $\Delta=3$ | $\Delta=5$ | Drop@5 |
|---|---:|---:|---:|---:|---:|
| Graph + hard filter + $L_{HAN}$ | 86.67 | 57.33 ± 3.06 | 42.00 ± 2.00 | 40.67 ± 2.31 | 46.00 pp |
| Graph + no second filter + $L_{HAN}$ | 85.33 | 61.33 ± 3.06 | 43.33 ± 1.15 | 33.33 ± 3.06 | 52.00 pp |

## 结果分析

- `graph_no_filter` 的低预算优势未随预算保持：相对 `graph_hard`，$\Delta=1,3,5$ 的攻击后 Micro-F1 差异依次为 4.00 pp、1.33 pp、-7.33 pp。
- 高预算下取消第二级硬过滤反而扩大下降，且 HetePRBCD 的最大配对损失超过预注册门槛；证据不支持用 `graph_no_filter` 替换当前实现。
- 后续敏感性实验固定使用 `graph_hard`；`han_semantic` 继续只作为研究开关，不混入同架构超参数比较。

## 版本判定

| Candidate | 最大 Clean 损失 | 最大 HetePRBCD 损失 | Adaptive@5 增益 | 通过 |
|---|---:|---:|---:|:---:|
| `graph_no_filter` | 0.83 pp | 3.82 pp | -7.33 pp | 否 |

- 冻结结论：`graph_hard`。
- 门槛：最大 clean 损失不超过 1.5 pp、最大 HetePRBCD 损失不超过 2 pp、Adaptive@5 攻击后 Micro-F1 至少提升 5 pp。
- 完整性：训练 12/12，自适应搜索 6/6，逻辑预算结果 18/18，问题数 0。
