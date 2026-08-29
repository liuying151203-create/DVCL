# DVCL 下一阶段开发与实验计划

本文档是当前唯一有效的后续实验计划。历史阶段记录见 `development-roadmap.md`，已完成结果见 `final-experiment-results.md`。

## 1. 当前基线

- 已完成 13 套协议、4124/4124 次物理运行，包括三数据集统一 11 模型 poisoning、RND、HG 迁移目标逃逸、多攻击种子复验、ACM 消融和正式模型自适应目标逃逸。
- HG Baseline 使用固定 artifact，不针对被评估模型优化。
- 阶段 A 已完成：统一 `adaptive_query` 已接入 11 个模型，并复用 165 个经哈希审计的 clean checkpoint。
- 阶段 B 已完成：候选筛选和 50 目标确认分别达到 12/12 与 48/48 次物理搜索，正式冻结每目标 64 条候选增边和 64 条候选删边；144/144 个 $\Delta\in\{1,3,5\}$ 预算评估审计通过。
- 阶段 C 已验收：`adaptive_target_evasion_v1` 在 ACM、DBLP、AMiner 上完成 11 模型正式矩阵，99/99 次物理搜索和 297/297 个预算评估通过审计，候选池哈希异常和无效结果均为 0；此前已完成的额外 DVCL 条件保留为补充结果，不进入正式统计。
- 阶段 D 已验收：15/15 个 clean checkpoint、60/60 次物理攻击评估和 90/90 个逻辑结果全部完成，强化审计问题数为 0；结果见 `dvcl-view-diagnosis-results.md`。
- 阶段 E 已验收：两个可靠性门控候选均未通过，论文模型冻结为 `concat`。
- 阶段 F1 已验收：AMiner 三种关系范围的 6 个 PRBCD/HetePRBCD artifact 和 12/12 次下游 Pilot 审计通过，但最佳 P–R 范围平均仅下降 0.07 pp，未达到 2 pp 扩展门槛。

## 2. 总体原则

1. 先建立足够强且公平的模型自适应攻击，再修改 DVCL，避免围绕弱攻击过拟合防御。
2. 每个模型独立选择攻击边，禁止复用针对 DVCL 优化的边评价其他基线。
3. 所有模型使用相同目标、候选采样 seed、候选池规模、预算与查询上限。
4. 主表继续只报告 Micro-F1；ASR、预算利用率、实际改边数和查询次数作为攻击有效性诊断。
5. 改进模型必须重新接受知道完整防御结构的自适应攻击，禁止只复用旧攻击。

### 执行与汇报规则

1. 运行超过 30 分钟的实验必须每半小时记录时间、完成/运行/失败数、剩余条件和预计完成时间；阶段 D 的结构化监控日志为 `outputs/logs/stage_d/progress-30min.jsonl`。
2. 任一阶段达到验收条件后必须暂停，不自动进入下一阶段；先提交完整性审计、核心结果、异常说明、结论边界和下一阶段建议，得到确认后再继续。
3. 失败任务必须保留原日志和状态，修复根因后利用 Runner 的断点语义重跑；禁止通过删除失败记录掩盖问题。
4. 单种子 Pilot 只用于机制筛选，不进入论文显著性结论；正式主张至少使用 3 个预注册配对重复。
5. 每阶段结束时同步更新结果文档、协议配置、复现命令和 Git 状态，论文表格继续只报告 Micro-F1。

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
| 配对重复 | $(s_a,s_t)=(1,1),(2,2),(3,3)$ |
| 目标 | 公共目标集；另统计 clean-correct 子集 |

正式矩阵为 $11\times3\times3\times3=297$ 个模型攻击评估单元，对应 99 次物理搜索；每次搜索复用 checkpoint 并沿同一贪心轨迹评估 $\Delta=1,3,5$。配对重复只需复用 $11\times3\times3=99$ 个 clean checkpoint，不重复训练模型。原 495 次全交叉设计因查询成本过高而取消，已生成的非配对 DVCL 条件仅作补充分析，不混入正式均值与标准差。

### 结果表

- 主表：每个模型的 clean target Micro-F1、$\Delta=1,3,5$ Micro-F1 和 Drop@5。
- 攻击审计表：ASR、实际改边数、预算利用率、查询次数和候选规模。
- 统计：按预注册的 $(s_a,s_t)$ 配对重复报告下降幅度、置信区间和 Holm 校正显著性；如出现异常条件，再定向追加重复。

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

### 已冻结的预检协议

| 维度 | 取值 |
|---|---|
| 数据集 | ACM、DBLP、AMiner |
| 模式 | `topo`、`feat`、`concat`、`gate`、`gated_concat` |
| 攻击 | HG Baseline 与模型自适应查询攻击 |
| 预算 | $\Delta=1,3,5$ |
| 预检种子 | $(s_a,s_t)=(1,1)$ |
| 候选池 | 每目标 64 条增边和 64 条删边 |

预检先训练 15 个 clean checkpoint，再执行 45 次 HG 迁移评估和 15 次自适应物理搜索；自适应搜索沿同一轨迹输出三个预算，因此共有 90 个逻辑攻击结果。诊断额外记录同一 checkpoint 下的分支置零 Micro-F1、真实类 margin、拓扑/特征 embedding 漂移、双视图预测分歧和 gate 权重，不改变训练目标和攻击选边。

```bash
source scripts/activate_gpu_env.sh
python scripts/run_suite.py \
  --config configs/protocols/dvcl_view_diagnosis_clean_pilot_v1.yaml
python scripts/run_suite.py \
  --config configs/protocols/dvcl_view_diagnosis_pilot_v1.yaml
python scripts/analyze_dvcl_view_diagnosis.py
```

预检完成后按以下门槛决定是否扩展到 3 个配对种子：DBLP 的 topology drift 和 margin 损失应明显高于 feature view，且 `feat` 或门控模式的攻击后 Micro-F1 优于 `concat`；ACM/AMiner 则重点排查零下降来自 feature view 稳定、融合分类边界冗余，还是有限候选池未覆盖有效结构边。若证据不一致，不进入阶段 E，而是先扩大候选池或补白盒梯度诊断。

### 验收结论

- 完整性：15/15 个 clean checkpoint、60/60 次物理攻击评估和 90/90 个逻辑结果完成；实验规格、输入路径、输入 SHA-256、victim checkpoint 和候选池哈希审计问题数均为 0。
- DBLP 自适应攻击在 $\Delta=5$ 下使 `topo`、`concat`、`gate`、`gated_concat` 分别下降 62、46、52、58 个百分点，而 `feat` 下降 0 个百分点，证明失效集中于拓扑视图。
- 现有门控没有识别拓扑失真：`gate` 和 `gated_concat` 的攻击后 Micro-F1 分别为 38 和 30，均低于 `concat` 的 42；其中 `gate` 相对 `concat` 为 $-4$ 个百分点，未达到预注册的 $+5$ 个百分点门槛。
- Feature embedding 漂移为 0 符合结构攻击定义，但 feature-only 的 DBLP clean Micro-F1 仅 79.98，不能直接用永久关闭拓扑视图替代可靠性建模。
- ACM/AMiner 上 `concat` 和 `gate` 的自适应 Drop@5 均为 0；该结论仅适用于当前 64+64 有限候选、结构修改威胁模型，不构成普遍鲁棒性证明。
- 本阶段 75 个 manifest 均来自 dirty worktree，结果只用于单种子机制筛选；进入论文统计的候选必须在冻结提交上重新训练并重新生成自适应攻击。

阶段 D 的预注册门槛判定为“不通过已有门控”。阶段 E 应先实现 `reliability_gate` 单种子机制 Pilot，而不是扩展 `gate`/`gated_concat` 到多种子，也不得直接把 `feat` 作为最终模型。

## 7. 阶段 E：DVCL 防御改进

候选方案按复杂度递增：

1. 使用已有 `gate`/`gated_concat` 替代固定 concat，建立直接对照。
2. 增加基于视图分歧、预测置信度和无 clean 参考的表征一致性代理量的可靠性感知门控 $\alpha_i$；clean/attacked embedding drift 只用于离线诊断，不作为主模型推理输入。
3. 训练时加入结构扰动，使门控学习在 topology 异常时降低其贡献。
4. 联合约束 clean 准确率、跨视图一致性和攻击条件稳定性，避免始终退化为 feature-only。

每个改进版本都必须在相同自适应攻击配置下重新生成攻击。验收时同时比较 clean Micro-F1、攻击后 Micro-F1、ASR、参数量、训练时间和查询成本。

### 阶段 E 决策门槛

1. 若阶段 D 中已有 `gate` 或 `gated_concat` 在 DBLP 自适应攻击下相对 `concat` 提升至少 5 个百分点，且三数据集 clean Micro-F1 损失不超过 1.5 个百分点，则先将该模式扩展到 3 个配对种子，不立即增加新模块。
2. 若已有门控不能满足门槛，则实现 `reliability_gate`：输入至少包含双视图预测分歧、分类置信度和无参考表征一致性代理量，并输出节点级拓扑权重 $\alpha_i$；禁止读取测试标签或 clean/attacked 成对 oracle。
3. 仅当 `reliability_gate` 仍不能满足门槛时，增加训练时结构扰动版本 `reliability_gate_aug`；不得同时引入多个无法独立归因的改动。
4. 候选模型必须重新生成针对自身的 64+64 自适应攻击，并与相同 seed 的 `concat`、`feat` 和最佳已有门控配对比较。
5. 正式验收要求：DBLP $\Delta=5$ Micro-F1 和 ASR 明显改善，ACM/AMiner 不出现超过 2 个百分点的攻击后退化，且门控不恒定退化为 feature-only。

阶段 E 先执行单种子机制 Pilot，暂停汇报后只对通过门槛的至多两个候选扩展到 $(s_a,s_t)=(1,1),(2,2),(3,3)$。正式矩阵规模在阶段 D 报告中冻结，禁止边跑边改评价口径。

### 阶段 E 验收结论

- `reliability_gate` 和 `reliability_gate_aug` 均完成 3/3 clean、12/12 物理攻击评估和 18/18 逻辑结果，输入、checkpoint、候选池与 manifest 审计问题均为 0。
- `reliability_gate` 的 DBLP 自适应 $\Delta=5$ Micro-F1 为 40，较 `concat` 低 2 pp；`reliability_gate_aug` 为 44，仅高 2 pp，未达到 5 pp 门槛。
- 增强版在 ACM 自适应攻击下较 `concat` 低 4 pp，超过允许的 2 pp 退化；两个候选均未满足三数据集门控非塌缩门槛。
- 阶段 E 判定为未通过，不扩展候选到多种子，也不继续基于单种子测试结果调参。后续冻结 `concat` 为论文主模型，并将 DBLP 自适应脆弱性写入局限性。
- 统一结果见 `docs/dvcl-stage-e-results.md`，机器可读审计见 `outputs/analysis/dvcl_reliability_gate_pilot_v1/` 和 `outputs/analysis/dvcl_reliability_gate_aug_pilot_v1/`。

## 8. 阶段 F：其余论文补充实验

1. AMiner PRBCD/HetePRBCD 增加 attack seed 2–3，并比较 P–A、P–R 和联合关系攻击，解决当前攻击偏弱问题。
2. 将组件消融扩展到 DBLP；资源允许时再补 AMiner。
3. 增加 $\lambda_h,\lambda_d,\tau,k,K$ 的敏感性实验。
4. 增加参数量、训练时间、推理时间和显存占用表。
5. 对最终主张重新计算显著性、下降幅度和排名，生成论文图表。

### 阶段 F 执行顺序

1. **F1 攻击可信度**：先补 AMiner PRBCD/HetePRBCD attack seed 2–3，关系类型按 P–A、P–R、联合攻击分别审计；若 surrogate 诊断仍显示攻击弱，先修攻击，不进入模型比较。
2. **F2 消融泛化**：将统一 `w/o` 组件消融扩展到 DBLP；AMiner 仅在主结论依赖该数据集时补充，避免无效全矩阵。
3. **F3 超参数敏感性**：围绕最终模型对 $\lambda_h,\lambda_d,\tau,k,K$ 做单因素实验，每次只改变一个参数；至少覆盖 clean 和一个最强攻击条件。
4. **F4 效率与资源**：报告参数量、训练时间、推理时间、攻击查询成本和峰值显存；时间使用相同硬件并预热后重复测量。
5. **F5 统计与图表**：对论文中的关键成对主张执行置信区间、Wilcoxon 和 Holm 校正，生成下降幅度、平均排名、攻击曲线和视图诊断图。

每个子阶段均先冻结协议和预期运行数，再执行输入审计、正式运行和完整性审计。F1–F5 不并行改变模型定义；阶段 E 一旦冻结，后续只补证据，不继续调参追逐测试结果。

### 阶段 F2 预注册协议

- 数据集：DBLP；变体为 Full DVCL、w/o Cross-view CL、w/o Feature View、w/o Topology View。
- 条件：clean、PRBCD/HetePRBCD 的 $r=5\%,15\%,25\%$；训练种子 $s_t=1,\ldots,5$，共 $4\times7\times5=140$ 次运行。
- 汇总：先在每个训练种子内跨扰动率计算攻击族平均，再报告五种子的 Micro-F1 均值与标准差；配对贡献定义为 Full DVCL 减去对应消融。
- 验收：140/140 次运行及输入哈希审计通过，全部 manifest 来自同一干净提交；Full DVCL 与 `dblp_poisoning_main_v1` 的 35 个对应条件逐项一致；w/o Topology View 在纯结构攻击下逐种子保持不变。
- 正式运行前必须先提交阶段 E、F1 和 F2 协议代码，禁止把 dirty Pilot manifest 写入论文消融表。

### 阶段 F1 验收结论

- 在 $r=15\%$、$(s_a,s_t)=(1,1)$ 下生成 P–A、P–R、P–A+P–R 三种等全局预算范围，共 6 个 PRBCD/HetePRBCD artifact；预算、关系、反向边和 provenance 验证全部通过。
- HeteroSAGE 与 DVCL 共完成 12/12 次下游训练，输入路径、artifact SHA-256 和 manifest 审计问题数为 0。
- 最佳公共范围为 P–R，但四个攻击/模型条件的平均 Micro-F1 下降仅 0.07 pp，远低于预注册的 2 pp 门槛；其他两种范围平均下降为负。
- P–A HetePRBCD 的 surrogate 下降 2.96 pp，但下游平均下降接近 0，表明弱点主要是代理攻击跨模型迁移不足，而不是预算未实现或仅攻击了错误关系。
- 按预注册规则停止 attack seed 2–3 和 360 次正式扩展；AMiner 现有 poisoning 结果保留为描述性证据，不用于宣称 DVCL 具备强 poisoning 鲁棒性。
- 结果见 `docs/aminer-poisoning-relation-pilot.md`，机器审计见 `outputs/analysis/aminer_poisoning_relation_pilot_v1/`。

## 9. 阶段 G：最终冻结

- 重新运行结果汇总和文档生成器。
- 在干净 GPU 环境执行完整 PyTest、环境审计和 12+ 新协议完整性审计。
- 冻结 Git 提交、环境版本、clean/split/attack artifact 哈希、checkpoint 哈希和结果哈希。
- 归档逐次运行结果、论文表格、图和完整复现命令。

### 首篇论文可投稿门槛

- 三数据集主表、poisoning/RND/HG/模型自适应攻击、关键消融、敏感性和效率实验均有完整审计；
- 论文核心结论至少有 3 个独立配对重复，效应量、95% CI 和多重比较校正齐全；
- DVCL 的鲁棒性表述明确限定威胁模型，不把有限候选零下降表述为普遍鲁棒；
- 若论文主张自适应防御，最终模型必须相对 `concat` 同时改善 clean 与最强攻击且不依赖单一数据集或 seed；当前候选未满足，因此首篇论文必须保留 `concat` 并明确不主张自适应鲁棒性；
- Git 提交、环境、artifact、checkpoint、结果和图表哈希全部冻结，复现命令在干净环境通过；
- 实验章节、方法章节、威胁模型、局限性和复现附录使用同一协议口径，不存在未解释的异常结果。

## 10. 开工顺序

阶段 A–E 和 F1 已全部验收。阶段 E 未产生通过门槛的新模型，已冻结 `concat` 为后续论文主模型；F1 证明当前 AMiner poisoning 的主要限制是跨模型迁移弱，并按门槛取消 360 次无效扩展。当前暂停在阶段边界，下一步经用户确认后进入 F2，将统一 `w/o` 组件消融扩展到 DBLP。
