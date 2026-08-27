# DVCL 下一阶段开发与实验计划

本文档是当前唯一有效的后续实验计划。历史阶段记录见 `development-roadmap.md`，已完成结果见 `final-experiment-results.md`。

## 1. 当前基线

- 已完成 12 套协议、4025/4025 次运行，包括三数据集统一 11 模型 poisoning、RND、HG 迁移目标逃逸、多攻击种子复验和 ACM 消融。
- HG Baseline 使用固定 artifact，不针对被评估模型优化。
- 阶段 A 已完成：统一 `adaptive_query` 已接入 11 个模型，并复用 165 个经哈希审计的 clean checkpoint。
- 阶段 B 已完成：候选筛选和 50 目标确认分别达到 12/12 与 48/48 次物理搜索，正式冻结每目标 64 条候选增边和 64 条候选删边；144/144 个 $\Delta\in\{1,3,5\}$ 预算评估审计通过。
- 阶段 C 已启动：`adaptive_target_evasion_v1` 在 ACM、DBLP、AMiner 上执行 11 模型正式矩阵，共 495 次物理搜索和 1485 个预算评估。

## 2. 总体原则

1. 先建立足够强且公平的模型自适应攻击，再修改 DVCL，避免围绕弱攻击过拟合防御。
2. 每个模型独立选择攻击边，禁止复用针对 DVCL 优化的边评价其他基线。
3. 所有模型使用相同目标、候选采样 seed、候选池规模、预算与查询上限。
4. 主表继续只报告 Micro-F1；ASR、预算利用率、实际改边数和查询次数作为攻击有效性诊断。
5. 改进模型必须重新接受知道完整防御结构的自适应攻击，禁止只复用旧攻击。

## 3. 阶段 A：通用模型自适应攻击

### 开发任务

- 将 `dvcl_adaptive_query` 重构为模型无关的 `adaptive_query` 请求协议。
- 从 DVCL adapter 中抽取统一攻击生成器，接入 HAN、HeteroSAGE、RoHe、HeteroGuard、FastRoHGCN、HGT、MAGNN、HeCo、SimpleHGN、HSeCo 和 DVCL。
- 每个 artifact 记录 victim model、训练种子、checkpoint 路径与哈希、候选 seed、候选池规模、查询次数和实际预算。
- 复用 clean checkpoint，避免每个 $\Delta$ 和 attack seed 重复训练模型。
- 增加 common-target 与 clean-correct 两种统计口径：前者用于同节点 Micro-F1，后者用于 ASR。

### 验收标准

- 相同 target/seed 下各模型候选池一致，但最终选择边允许不同。
- 自适应评估阶段无 optimizer step，clean checkpoint 哈希保持不变。
- 每个目标实际改边数不超过 $\Delta$，反向边、关系和 artifact 验证全部通过。
- 至少两个模型在同一候选池下生成不同攻击记录，证明攻击确实针对模型优化。

## 4. 阶段 B：攻击强度 Pilot

### 分层 Pilot

50 目标基准显示，HAN/ACM 的 16+16 候选在 $\Delta=1$ 和 $\Delta=5$ 下分别约需 117 秒和 292 秒。为避免直接执行 432 单元全因子造成无效计算，阶段 B 分两步执行。

第一步为候选规模筛选：

| 维度 | 取值 |
|---|---|
| 数据集 | ACM、DBLP |
| 模型 | HAN、DVCL |
| 目标数 | 10 |
| 候选池 | 16+16、64+64、128+128 |
| 预算 | $\Delta=5$ |
| 训练种子 | 1 |
| 候选 seed | 1 |

筛选共 12 个单元。第二步仅保留筛选出的最小稳定候选规模，并恢复 50 个公共目标：

| 维度 | 取值 |
|---|---|
| 数据集 | ACM、DBLP |
| 模型 | HAN、HeteroGuard、HSeCo、DVCL |
| 候选池 | 筛选出的规模；必要时增加相邻更大规模复核 |
| 预算 | $\Delta=1,3,5$ |
| 训练种子 | 1–2 |
| 候选 seed | 1–3 |

单候选规模确认矩阵为 144 个单元；只有稳定性判据不通过时才执行原 432 单元全因子配置。

### 选择规则

- 优先选择 ASR 和 Micro-F1 下降随候选池扩大趋于稳定的最小候选规模。
- 若预算利用率持续偏低，继续扩大候选池或实现完整/分块候选搜索。
- 若 $\Delta=3$ 与 $\Delta=5$ 长期平台，检查目标已误分类、候选耗尽和贪心局部最优三种原因。
- Pilot 输出独立协议，不覆盖当前正式结果。

## 5. 阶段 C：十一模型正式自适应矩阵

### 正式矩阵

| 维度 | 取值 |
|---|---|
| 数据集 | ACM、DBLP、AMiner |
| 模型 | 统一 11 模型 |
| 预算 | $\Delta=1,3,5$ |
| 训练种子 | 1–5 |
| 候选 seed | 1–3 |
| 目标 | 公共目标集；另统计 clean-correct 子集 |

完整矩阵为 $11\times3\times3\times5\times3=1485$ 个模型攻击评估单元。它们应复用 $11\times3\times5=165$ 个 clean checkpoint，而不是重复完成 1485 次训练。

### 结果表

- 主表：每个模型的 clean target Micro-F1、$\Delta=1,3,5$ Micro-F1 和 Drop@5。
- 攻击审计表：ASR、实际改边数、预算利用率、查询次数和候选规模。
- 统计：按训练种子与候选 seed 配对，报告下降幅度、置信区间和 Holm 校正显著性。

## 6. 阶段 D：DVCL 失效诊断

在修改模型前比较以下现有模式：

- `topo`：仅拓扑视图；
- `feat`：仅特征视图；
- `concat`：当前默认拼接；
- `gate`：加权求和；
- `gated_concat`：加权拼接。

对 clean、HG 迁移攻击和正式自适应攻击记录：

- 两个视图各自的目标 Micro-F1 与分类间隔；
- topology/feature embedding 的攻击前后漂移；
- 两视图预测分歧；
- gate 权重分布及其与攻击成功的关系。

只有当结果证明 topology view 在攻击下明显失真而 feature view 相对稳定时，才进入动态特征权重防御。

## 7. 阶段 E：DVCL 防御改进

候选方案按复杂度递增：

1. 使用已有 `gate`/`gated_concat` 替代固定 concat，建立直接对照。
2. 增加基于视图分歧、预测置信度和 topology 漂移的可靠性感知门控 $\alpha_i$。
3. 训练时加入结构扰动，使门控学习在 topology 异常时降低其贡献。
4. 联合约束 clean 准确率、跨视图一致性和攻击条件稳定性，避免始终退化为 feature-only。

每个改进版本都必须在相同自适应攻击配置下重新生成攻击。验收时同时比较 clean Micro-F1、攻击后 Micro-F1、ASR、参数量、训练时间和查询成本。

## 8. 阶段 F：其余论文补充实验

1. AMiner PRBCD/HetePRBCD 增加 attack seed 2–3，并比较 P–A、P–R 和联合关系攻击，解决当前攻击偏弱问题。
2. 将组件消融扩展到 DBLP；资源允许时再补 AMiner。
3. 增加 $\lambda_h,\lambda_d,\tau,k,K$ 的敏感性实验。
4. 增加参数量、训练时间、推理时间和显存占用表。
5. 对最终主张重新计算显著性、下降幅度和排名，生成论文图表。

## 9. 阶段 G：最终冻结

- 重新运行结果汇总和文档生成器。
- 在干净 GPU 环境执行完整 PyTest、环境审计和 12+ 新协议完整性审计。
- 冻结 Git 提交、环境版本、clean/split/attack artifact 哈希、checkpoint 哈希和结果哈希。
- 归档逐次运行结果、论文表格、图和完整复现命令。

## 10. 开工顺序

阶段 A、B 已验收，当前执行阶段 C。正式矩阵完整审计后，先依据阶段 D 诊断解释模型差异，再决定是否进入阶段 E 的 DVCL 改进；不得在完整自适应基线形成前提前调整防御结构。
