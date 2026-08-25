# 攻击威胁模型审计报告

审计日期：2026-08-05；状态更新：2026-08-24。

HG Baseline target evasion 的 artifact、验证、clean-train/attacked-test Trainer 分支和
三数据集正式矩阵均已完成；ACM/DBLP 共 330 次，AMiner 共 165 次。另已完成 DVCL
有限候选模型自适应目标攻击 30 次。当前结果和限制以 `docs/final-experiment-results.md` 为准。

## 1. 结论

当前项目已经完成的 PRBCD/HetePRBCD 实验属于**全局结构投毒攻击
（global poisoning）**，不属于测试时逃逸攻击（evasion）。非 clean 条件下，HAN、
HeteroSAGE、HSeCo 和 DVCL 都直接在扰动图上训练、验证并测试。

这与参考论文的明确设定一致：PRBCD 和 HetePRBCD 是 poisoning attacks，HG
Baseline 是 evasion attack。因此，现有全局攻击实验的威胁模型没有配错，也不应改成
evasion 后替代现有结果。

需要区分旧 HSeCo 工程中的两套协议：

1. `our_global.py`、`hete_global.py` 和 `our_dvcl.py` 先加载扰动图，再训练模型，
   对应全局 poisoning；当前 PRBCD/HetePRBCD 主实验复现的是这套语义。
2. `our_local.py` 和 `hete_local.py` 先在 clean 图上训练并恢复 checkpoint，再逐目标
   替换测试图、固定训练结束时的语义注意力并仅执行前向推理，对应 Atk_RoHe 目标
   逃逸攻击。

因此，当前实现并非把旧 HSeCo 的**全局攻击协议**误写成 poisoning。首次审计时缺失的
第二套 **HG Baseline/Atk_RoHe 目标 evasion 协议**现已完成工程实现。
ACM/DBLP 的 11 模型正式矩阵已完成 330/330 次并通过完整性审计，Micro-F1 结果见
`docs/target-evasion-results.md`。

威胁模型判定正确不代表攻击生成参数已经符合论文。首次审计发现旧生成包装脚本没有
传入 `--constrained`，HetePRBCD 也没有传入 `--biased`。这些旧结果现已被修正 artifact
和 ACM/DBLP 共 580 次正式主实验、基线与消融结果取代；clean 结果不受影响。

## 2. 参考论文协议

论文把三种攻击分为两个互不替代的实验族：

| 攻击 | 威胁模型 | 扰动单位 | 当前状态 |
|---|---|---|---|
| PRBCD | Poisoning | 全局扰动率 5%–25% | 修正配置正式实验已完成 |
| HetePRBCD | Poisoning | 全局扰动率 5%–25% | 修正配置正式实验已完成 |
| HG Baseline | Evasion | 每个目标节点的扰动预算 $\Delta\in\{1,3,5\}$ | ACM/DBLP/AMiner 已完成 |
| DVCL Adaptive | Evasion | 每个目标节点的最大预算 $\Delta\in\{1,3,5\}$ | ACM/DBLP 有限候选版本已完成 |

HG Baseline 在 ACM 和 DBLP 上攻击 Paper–Author（P–A）边，在 AMiner 上攻击
Paper–Reference（P–R）边。三套数据的目标逃逸 artifact 均已完成并通过验证。

## 3. 代码证据

| 检查项 | 当前行为 | 判定 |
|---|---|---|
| 协议配置 | `dvcl_main.yaml`、`baseline_main.yaml` 明确设置 `threat_model: poisoning`、`scope: global` | 显式声明全局 poisoning |
| Runner | `run_experiment.py` 校验 artifact 协议，并把 `threat_model`、`scope` 传入原生 trainer | target evasion 进入独立 clean-train/attacked-test 分支 |
| HAN | 训练前用 `attack.perturbed_hete_adjs` 构建全部元路径图 | 在攻击图上训练 |
| HeteroSAGE | 训练前用扰动邻接构建异构图 | 在攻击图上训练 |
| HSeCo | 训练前基于扰动邻接生成 transition、净化视图和语义拓扑 | 在攻击图上训练 |
| DVCL | 训练前基于扰动邻接生成拓扑视图；优化过程持续使用该视图 | 在攻击图上训练 |
| 旧 global 入口 | 先调用 `load_perturbed_data`，后执行训练循环 | poisoning |
| 旧 local 入口 | clean 图训练和 checkpoint 恢复完成后，逐目标构建攻击图并在 `torch.no_grad()` 下推理 | target evasion |

生成脚本现已修正为 PRBCD 显式启用 semantic constraints，HetePRBCD 显式启用
semantic constraints 与 biased sampling。新源文件会保存 `constrained`、`biased`、
`lambda`、budget、seed 和训练轮数等 provenance，并先进入独立 pilot 目录，不覆盖旧
artifact。

`adaptive: false` 与 poisoning/evasion 是两个独立维度。它只表示攻击没有针对每个
被评估模型重新优化；同一个非自适应攻击既可以在训练前注入形成 poisoning，也可以
只在测试时应用形成 transfer evasion。`scope: global` 只描述扰动覆盖范围，同样不能
决定攻击发生在训练阶段还是测试阶段。

## 4. 已有结果的含义

正式完成的攻击运行均在 manifest 中记录为 `threat_model: poisoning`：

| 结果集 | Clean | Poisoning 攻击 | 合计 |
|---|---:|---:|---:|
| `dvcl_main` 主实验（HSeCo、DVCL） | 20 | 200 | 220 |
| `dvcl_main` ACM 消融 | 20 | 120 | 140 |
| `baseline_main`（HAN、HeteroSAGE） | 20 | 200 | 220 |
| **合计** | **60** | **520** | **580** |

表中 580 次矩阵随后已使用修正 artifact 全部重跑，正式数字见 ACM、DBLP 和跨数据集
结果文档。ACM 上下降较小可能来自模型对扰动图的重新适应、早停选择和 DVCL 的固定
特征视图；这也不能据此推断 HG Baseline evasion 下的鲁棒性。

ACM/DBLP/AMiner 的 `scope: target` 十一模型矩阵已完成。HSeCo/DVCL
目标前向固定训练结束时的语义注意力，DVCL 同时复用不受结构扰动影响的 feature view，
只重建被目标边修改影响的 topology view。

## 5. 复现 HG Baseline 的正确流程

应保留现有 PRBCD/HetePRBCD 结果，并明确标注为 global poisoning；另行实现和运行
HG Baseline/Atk_RoHe target evasion。两类结果应分别成表，不能合并为同一 Attack
Average。目标扰动预算严格使用 $\Delta\in\{1,3,5\}$。

目标 evasion 的执行顺序应为：

1. 仅用 clean 图完成训练、验证、早停和 checkpoint 恢复。
2. 在相同 test mask 中确定目标节点，并保存 clean 目标节点预测。
3. 对每个目标节点加载其增删边集合，临时构造目标攻击图。
4. 冻结参数且不执行任何 optimizer step，只在攻击图上前向推理目标节点。
5. 汇总目标节点 clean Micro-F1、attacked Micro-F1、绝对下降和攻击成功率。

## 6. 可选的全局 evasion 扩展

应新增独立协议，例如 `dvcl_global_evasion`，不要覆盖现有 poisoning 产物。正确语义是
clean 图训练、clean checkpoint 恢复、全局扰动图测试。现有攻击 artifact 可作为
`adaptive: false` 的 transfer evasion 输入复用；若要做自适应攻击，则需针对每个
模型和 seed 重新生成攻击。

当前 DVCL 自适应扩展已经按 checkpoint 独立生成攻击，但只查询每目标最多 16 条候选
增边与 16 条候选删边，并且 $\Delta$ 是最大预算而非强制预算。该结果属于有限候选、
score-based 的模型自适应攻击，不应表述为完整候选空间的最强梯度白盒攻击。

四个 trainer 都需要显式接收 `threat_model`，并分别构建训练图和测试图。输出应同时
保存 clean-test 与 attacked-test 指标，汇总器在攻击行使用 attacked-test 指标。
若全局攻击全部改为 evasion，现有 520 个 poisoning 攻击运行需要在新输出目录重跑，
60 个 clean 运行仍可作为参考。

全局 evasion 不属于上述论文三种攻击的主协议，只能作为额外实验，不能替代
PRBCD/HetePRBCD poisoning 或 HG Baseline target evasion。

## 7. 修复优先级与验收标准

1. **阻断错误配置（已完成）**：Runner 校验 artifact 的 threat model、scope 和
   adaptive 声明，并拒绝尚未支持的 global evasion。
2. **实现目标 evasion（已完成）**：增加目标节点和逐目标攻击 artifact，按
   $\Delta\in\{1,3,5\}$ 复现旧 local 入口的 clean-train/attacked-test 流程。
3. **实现全局 evasion**：仅在额外研究需要时，为四个原生 trainer 增加双图评估路径。
4. **补充协议测试**：通过 mock/diagnostics 证明 optimizer step 只访问 clean 图，攻击
   图只在 checkpoint 恢复后的 `eval()` 与 `no_grad()` 阶段使用。
5. **结果隔离**：poisoning、global evasion、target evasion 使用不同 protocol 与输出
   目录，并在表题中标明 `scope`、`adaptive` 和训练/测试图。

验收时至少检查：manifest 威胁模型与实际执行分支一致；evasion 的训练 history 与同
seed clean 运行一致；checkpoint 哈希与对应 clean 运行一致；评估阶段无参数更新；目标
攻击只统计约定目标节点。

## 8. 修正攻击 pilot

修正后的 DBLP 攻击先生成到 `outputs/pilots/attack_protocol`，不会覆盖当前
`data/attacks`。GPU 驱动恢复后执行：

```bash
source scripts/activate_gpu_env.sh
python scripts/prepare_dblp_attack_pilot.py --gpu-id 0
python scripts/run_suite.py \
  --config configs/protocols/dblp_attack_protocol_pilot.yaml \
  --python-bin "$CONDA_PREFIX/bin/python"
python scripts/summarize_results.py \
  --run-root outputs/runs/dblp_attack_protocol_pilot_v2 \
  --output-dir outputs/summaries/dblp_attack_protocol_pilot_v2
```

该 pilot 共 16 次运行：4 个模型、2 种攻击、2 个扰动率、1 个训练 seed。只有当新
artifact provenance、预算、反向边和 split 扰动报告均通过，且 HAN 不再出现未经解释
的 epoch-0 类别塌缩后，才扩展到全部扰动率和 5 个训练 seed。
