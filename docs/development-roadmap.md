# DVCL 后续开发与实验路线

## 1. 文档目标

本文档基于 2026-08-03 的仓库状态，说明阶段 1–3 完成后的后续开发、验证和正式实验计划。重点是：

- 固化可复现运行环境；
- 补齐 DBLP 数据和攻击产物；
- 扩大旧 HSeCo 与原生实现的 golden 对照覆盖；
- 按先试运行、后主实验、再消融的顺序生成正式结果；
- 为最终论文表格、结果审计和代码发布建立明确验收门槛。

本路线中的“完成”不仅表示代码能够运行，还要求输入 artifact、随机种子、软件环境、checkpoint 和输出指标均可追溯。

## 2. 当前基线

### 2.1 已完成的工程阶段

阶段 1：原生实验产物流水线。

- ACM/DBLP clean artifact 生成接口；
- 固定 split 生成与旧 split 导入；
- RND 生成；
- PRBCD/HetePRBCD 导入、差分统计和攻击验证；
- 旧 HSeCo artifact 转换；
- 数据、划分和攻击脚本不再依赖旧 HSeCo 工程。

阶段 2：HSeCo 和 DVCL 原生训练实现。

- HSeCo 等效语义净化、语义注意力和节点聚合；
- DVCL 双视图模型；
- 原生 HSeCo/DVCL 训练适配器；
- 旧式早停和 checkpoint 语义；
- 统一协议下的完整模型恢复；
- artifact 和指标等效性比较工具。

阶段 3：可复现实验编排。

- 原生模型注册器；
- 单次实验 Runner；
- 独立 split、attack 和 train seed；
- 完成实验自动跳过和 `--force` 重跑；
- 失败状态和 traceback 记录；
- manifest、metrics、history 和 checkpoint 输出；
- 主实验与消融 suite；
- 均值、标准差和 Attack Average 汇总。

当前主实验矩阵为 220 次运行，消融实验矩阵为 140 次运行。

### 2.2 当前环境和验证状态

已建立本地 CPU 验证环境和项目内 CUDA 12.1 环境：

```text
Python             3.11.15
Torch              2.1.2+cpu
TorchVision        0.16.2+cpu
TorchAudio         2.1.2+cpu
DGL                1.1.3
Torch Geometric    2.5.3
NumPy              1.26.4
SciPy              1.11.4
Pandas             2.2.2
scikit-learn       1.4.2
PyTest             8.4.2
```

GPU 环境位于 `.conda/dvcl-cu121-py39`，使用 Python 3.9.25、Torch
2.1.2+cu121、DGL 1.1.3+cu121 和 Torch Geometric 2.5.3。CUDA wheel
及动态库导入已通过。当前开发机配置 7 张 Tesla V100-PCIE-32GB，
在沙箱外执行的 GPU 张量前向与反向 smoke 已通过。

已通过：

- 完整 PyTest，27 项测试全部通过；
- 所有 Python 文件静态编译；
- `DVCL experiment contracts: passed`；
- `pip check`；
- `git diff --check`；
- Torch、DGL 和 PyG 真实导入；
- DGL 模型前向和 ACM HSeCo 完整训练；
- Runner 的 manifest、history、checkpoint、metrics 和 status 输出。

### 2.3 当前 ACM artifact 状态

已生成：

- ACM clean artifact；
- `paper_seed_1` split，806/402/2817；
- random seed 1 split，402/402/3221；
- 旧 HSeCo split 导入产物，808/401/2816；
- PRBCD 5%、10%、15%、20%、25%；
- HetePRBCD 5%、10%、15%、20%、25%。

主协议使用的 10 个攻击 artifact 已绑定 `paper_seed_1`，并全部通过攻击验证。

### 2.4 当前 DBLP artifact 状态

已生成并验收：

- DBLP clean artifact，节点数与关系边数和旧 HSeCo 数据精确一致；
- `paper_seed_1` 全局实验 split，405/405/3247；
- PRBCD 5%、10%、15%、20%、25%；
- HetePRBCD 5%、10%、15%、20%、25%；
- 攻击源特征、标签、train/val/test mask 强制一致性检查；
- 去除反向边重复计数后的全局攻击预算统计；
- 主协议 24 个唯一输入的哈希、身份和攻击验证，24/24 通过。

DBLP 全局实验沿用旧 HSeCo PRBCD/HetePRBCD 源文件绑定的随机 10/10/80
划分。PyG 官方 400/400/3257 mask 属于另一实验口径，不能与当前全局攻击源混用。

### 2.5 当前 golden 对照状态

Artifact 层已完成：

- ACM clean 精确一致；
- random seed 1 split 精确一致；
- 5 个 PRBCD artifact 精确一致；
- 5 个 HetePRBCD artifact 精确一致。

模型层已完成 ACM clean、train seed 1 的同环境严格对照：

- 旧 HSeCo 和原生 HSeCo 共比较 142 个验证 epoch；
- Micro-F1 和 Macro-F1 轨迹最大差异均为 0；
- 最佳 epoch 均为 39；
- 停止 epoch 均为 141；
- 最终 Accuracy、Micro-F1 和 Macro-F1 差异均为 0。

该结果证明在相同 CPU、依赖、artifact 和随机种子下，原生 HSeCo clean 训练行为与旧实现一致。

模型层已完成 DBLP clean、train seed 1 的同环境 GPU 严格对照：

- 旧 HSeCo 和原生 HSeCo 共比较 200 个验证 epoch；
- 验证 Micro-F1 和 Macro-F1 轨迹最大差异均为 0；
- 最终 Accuracy、Micro-F1 和 Macro-F1 差异均为 0；
- reference 命令显式冻结 `negative_noise_rate=0.01`，避免旧仓库 YAML 的
  `0.001` 默认覆盖统一协议配置。

### 2.6 尚未完成

- GPU 环境、张量 smoke 和 ACM 单 epoch GPU Runner 已验收；
- HSeCo 攻击场景模型 golden 尚未完成；
- DVCL 模型 golden 尚未完成；
- ACM 110 次主实验已经完成，DBLP 110 次主实验尚未运行；
- ACM 140 次消融实验已经完成；
- ACM 结果表已生成，DBLP 与双数据集论文表尚未生成；
- 其他基线和目标攻击尚未进入当前统一原生协议。

## 3. 执行原则

后续阶段统一遵守以下原则：

1. 先固定环境，再生成正式结果。
2. 训练只读取冻结 artifact，不在训练过程中重新划分或生成攻击。
3. split、attack 和 train seed 保持独立。
4. 同一协议下所有模型读取相同 clean、split 和 attack artifact。
5. golden 对照和正式实验分目录保存，禁止覆盖主协议 artifact。
6. CPU/GPU 或依赖版本不同的结果不能作为严格零差异 golden。
7. 失败运行通过 Runner 原命令重跑，不手工补写 metrics。
8. `outputs/` 不提交 Git，但正式结果必须独立归档和备份。

## 4. 阶段 4：正式环境固化

### 4.0 实施状态

阶段 4 的工程实现和项目内 GPU 环境组装已经完成：

- CPU/CUDA 12.1 两套固定 requirements；
- 项目内 `.conda/dvcl-cu121-py39` CUDA 环境；
- GPU 环境激活和动态库路径配置脚本；
- CPU/GPU 环境预检 CLI；
- Torch、DGL 和 PyG 前向与反向 smoke；
- CUDA 请求严格失败策略；
- manifest schema 2 环境审计；
- 正式环境安装说明。

当前开发机宿主环境具有可用 NVIDIA 驱动和 7 张 Tesla V100。
CPU profile、CUDA wheel、DGL/PyG 导入以及 GPU profile 张量正向/反向
已完成验收。Codex 沙箱默认不映射 GPU 设备，所以 GPU 命令需在
普通主机 shell 或授权的沙箱外执行。ACM HSeCo 单 epoch GPU Runner
已完成，并正确生成 checkpoint、history、metrics、status 和 manifest。

### 4.1 目标

建立可用于正式实验的 Python 3.9–3.11 GPU 环境，并将软件版本、CUDA 状态和设备信息纳入实验审计。

建议基线：

```text
Python 3.9–3.11
Torch 2.1.2
CUDA 12.1
DGL 1.1.3+cu121
Torch Geometric 2.5.3
```

### 4.2 开发任务

- 增加环境预检脚本，检查 Python、Torch、DGL、PyG、CUDA 和 GPU；
- 明确区分 CPU 验证环境和 GPU 正式环境；
- 固化 GPU wheel 安装说明；
- 导出环境快照或 lock 文件；
- 确保 manifest 记录 CUDA、GPU、Torch、DGL 和 PyG 版本；
- 对 CUDA 不可用的正式协议选择失败退出，而不是静默回退 CPU。

### 4.3 验收标准

- `torch.cuda.is_available()` 为 `True`；
- Torch、DGL 和 PyG 能在目标 GPU 上完成前向与反向；
- 一个 ACM HSeCo clean 单 epoch Runner 成功；
- manifest 中能够识别设备和核心依赖版本；
- CPU 和 GPU 输出目录或 protocol 名称不会混淆。

### 4.4 交付物

- 环境安装说明；
- 环境预检脚本；
- GPU 环境激活脚本；
- GPU 环境快照；
- ACM 单 epoch GPU smoke 运行。

## 5. 阶段 5：DBLP 数据与攻击产物

### 5.0 实施状态

阶段 5 已完成。DBLP clean、全局实验 split 和 10 个攻击 artifact 已全部生成，
并通过统一协议输入审计。新增 `scripts/check_protocol_inputs.py`，可按协议 YAML
检查所有输入文件、哈希、数据身份、split 绑定、反向边和全局扰动预算。

### 5.1 目标

补齐 DBLP clean、固定 split 和全部攻击条件，使两个数据集都满足主实验输入要求。

### 5.2 开发任务

- 下载并生成 DBLP clean artifact；
- 核对节点类型、预测节点类型、关系方向、特征和标签；
- 生成 `paper_seed_1` split；
- 明确 DBLP paper 协议的实际比例和随机化方式；
- 获取或生成 DBLP PRBCD/HetePRBCD 5%–25% 源文件；
- 将攻击导入原生格式；
- 验证关系集合、边方向、反向边、预算和 split 身份；
- 为 DBLP 增加统计基线和 golden 报告。

### 5.3 验收标准

- clean、split 和 10 个攻击 artifact 全部存在；
- 所有 artifact schema 和哈希可读取；
- 所有攻击验证结果为 `ok: true`；
- 主实验 dry-run 不再报告 DBLP 输入缺失；
- DBLP clean 和 split 至少完成一次独立复核。

### 5.4 交付物

- DBLP clean artifact；
- DBLP `paper_seed_1` split；
- DBLP 10 个攻击 artifact；
- DBLP artifact 验证报告。

## 6. 阶段 6：扩大 Golden 覆盖

### 6.0 实施状态

- 新增旧训练 stdout 与原生 `history.csv`/`metrics.json` 的自动比较入口；
- DBLP HSeCo clean、train seed 1、同 V100 环境已达到 200 epochs 严格零差异；
- ACM 攻击条件、DBLP 攻击条件和 DVCL golden 仍待扩展。

### 6.1 目标

从 ACM HSeCo clean 的单条件严格对照，扩大到攻击条件、DVCL 和 DBLP。

### 6.2 推荐矩阵

HSeCo：

- ACM clean；
- ACM PRBCD 5% 和 25%；
- ACM HetePRBCD 5% 和 25%；
- DBLP clean；
- DBLP PRBCD/HetePRBCD 代表条件。

DVCL：

- ACM clean；
- ACM PRBCD 5%；
- ACM HetePRBCD 5%；
- DBLP clean；
- 至少一个 DBLP 攻击条件。

首轮每个条件使用 train seed 1。首轮通过后，再选择关键条件扩展到 train seed 1–3。

### 6.3 对照内容

- 输入 artifact 哈希；
- 每个 epoch 的验证指标；
- 语义注意力；
- 最佳 epoch 和停止 epoch；
- checkpoint 语义；
- 最终 Accuracy、Micro-F1 和 Macro-F1；
- 攻击验证报告。

### 6.4 对照标准

同设备、同依赖和同 seed：

- 优先要求离散 artifact 完全一致；
- clean 训练优先要求严格一致；
- 若底层稀疏算子存在非确定性，记录最大绝对差异并解释来源。

跨设备或跨依赖版本：

- 不要求逐 epoch 完全相同；
- 最终指标使用明确容差；
- 默认指标容差为 0.005；
- 不能用跨设备容差结果替代同环境严格 golden。

### 6.5 开发任务

- 增加可批量运行 golden 的配置或脚本；
- golden artifact 使用独立输出路径；
- 输出单次报告和汇总报告；
- 报告记录 reference/current 环境；
- 对失败条件保留差异明细。

### 6.6 验收标准

- 推荐矩阵全部有报告；
- 所有 artifact 对照通过；
- 模型差异均在预定义标准内；
- 不存在未解释的 split、checkpoint 或设备差异。

## 7. 阶段 7：GPU Pilot 与主实验

### 7.0 实施状态

- ACM 110 次主实验已完成并汇总；
- DBLP HSeCo 与 DVCL 在 clean、PRBCD 5% 和 HetePRBCD 5% 上的 6 个单 epoch
  GPU pilot 已完成；
- 6 个 pilot 均使用 Tesla V100 和 `cuda:0`，status 为 `completed`；
- 两个攻击条件的 `attack_verification.json` 与全局预算检查均通过；
- DBLP 正式 110 次主实验尚未启动。

### 7.1 目标

先通过小规模 GPU pilot 验证资源、耗时、显存和恢复机制，再执行 220 次主实验。

### 7.2 Pilot 顺序

建议先运行：

1. ACM HSeCo clean，train seed 1；
2. ACM DVCL clean，train seed 1；
3. ACM HSeCo PRBCD 5%，train seed 1；
4. ACM DVCL PRBCD 5%，train seed 1；
5. ACM HSeCo HetePRBCD 5%，train seed 1；
6. ACM DVCL HetePRBCD 5%，train seed 1。

### 7.3 Pilot 检查项

- 峰值显存；
- 单 epoch 和单次实验耗时；
- checkpoint 大小；
- early stopping 是否正常；
- history 是否逐 epoch 写入；
- 中断后是否能跳过已完成运行；
- 失败状态是否包含 traceback；
- 汇总脚本是否能读取 pilot 结果。

### 7.4 主实验顺序

建议分批执行：

1. ACM 主实验 110 次；
2. 汇总 ACM 并检查异常 seed；
3. DBLP 主实验 110 次；
4. 汇总完整 220 次；
5. 对失败或异常运行进行定点重跑。

### 7.5 验收标准

- 220 个预期 run 目录全部存在；
- 所有 `status.json` 均为 `completed`；
- 每个条件包含 5 个 train seed；
- 同条件输入哈希一致；
- 没有使用错误 split 的攻击 artifact；
- 均值、标准差和 Attack Average 可完整生成。

## 8. 阶段 8：消融实验

### 8.0 实施状态

ACM 140 次 DVCL 组件消融已全部完成，所有条件包含 5 个训练种子，完整 DVCL
与主实验重叠条件结果一致。DBLP 消融不在当前 140 次 suite 范围内。

### 8.1 目标

在主实验协议稳定后执行 140 次 DVCL 消融，避免在主模型仍变化时提前生成无效结果。

### 8.2 当前消融范围

- 完整 DVCL；
- 移除对比损失；
- 仅拓扑视图；
- 仅特征视图。

### 8.3 执行要求

- 所有变体共用主实验 clean、split 和 attack artifact；
- 保持相同训练预算、patience 和 train seed；
- 只修改变体定义的组件；
- 每个运行的 manifest 明确记录 variant；
- 先对每个变体执行一个 clean smoke。

### 8.4 验收标准

- 140 个预期运行全部完成；
- 每个变体的运行数量与 suite 展开一致；
- 消融结果可以按数据集、攻击、扰动率和变体汇总；
- 完整 DVCL 的消融基线与主实验对应结果一致。

## 9. 阶段 9：汇总、审计与发布

### 9.0 实施状态

ACM 主实验、Attack Average 和组件消融已汇总到
`docs/acm-experiment-results.md`。DBLP 结果、双数据集论文表、显著性检验和
最终发布归档仍待完成。

### 9.1 目标

把原始运行结果转化为可审计、可引用和可归档的最终实验材料。

### 9.2 开发任务

- 运行结果汇总；
- 检查缺失条件和重复条件；
- 输出均值、标准差和 Attack Average；
- 生成主实验表和消融表；
- 记录异常运行及处理方式；
- 归档环境、配置、Git commit、artifact 哈希和原始 outputs；
- 更新 README 和实验文档；
- 增加最终复现命令清单。

### 9.3 验收标准

- 主实验 220 次和消融 140 次均完整；
- 汇总表能追溯到每个 `metrics.json`；
- 每个结果能追溯到 manifest 和输入哈希；
- 不混用论文原始数字、旧协议数字和统一协议结果；
- 发布包包含环境、配置、代码 commit 和结果摘要。

## 10. 推荐命令顺序

### 10.1 提交前验证

```bash
python -m pytest -ra
python -m compileall -q src scripts tests
python scripts/check_contracts.py
git diff --check
```

### 10.2 DBLP 准备

```bash
python scripts/prepare_dataset.py --dataset dblp
python scripts/generate_split.py --dataset dblp --seed 1 --protocol paper
```

攻击导入示例：

```bash
python scripts/generate_attack.py \
  --dataset dblp \
  --split paper_seed_1 \
  --attack prbcd \
  --attack-rate 5 \
  --seed 1 \
  --mode import \
  --source-file PATH_TO_DBLP_PRBCD_5
```

### 10.3 矩阵检查

```bash
python scripts/run_suite.py \
  --config configs/protocols/dvcl_main.yaml \
  --dry-run

python scripts/run_suite.py \
  --config configs/suites/dvcl_component_ablation.yaml \
  --dry-run
```

### 10.4 主实验

```bash
python scripts/run_suite.py \
  --config configs/protocols/dvcl_main.yaml \
  --continue-on-error
```

### 10.5 消融实验

```bash
python scripts/run_suite.py \
  --config configs/suites/dvcl_component_ablation.yaml \
  --continue-on-error
```

### 10.6 汇总

```bash
python scripts/summarize_results.py
```

## 11. 风险与应对

### 11.1 GPU 非确定性

风险：CUDA 稀疏算子和聚合顺序可能导致逐 epoch 数值差异。

应对：

- 同环境严格 golden；
- 跨环境使用明确容差；
- 保存环境和设备信息；
- 不把跨设备差异直接判定为实现错误。

### 11.2 攻击路径不包含 split

风险：当前攻击路径主要由 dataset、attack、rate 和 attack seed 决定，不同 split 的攻击 artifact 可能发生覆盖。

应对：

- 主协议路径只保存 `paper_seed_1` 对应攻击；
- golden 或临时 split 必须使用显式 `--output`；
- 每次训练前依赖 `verify_attack` 检查 split identity；
- 后续评估是否将 split_name 纳入攻击路径。

### 11.3 DBLP 攻击源不完整

风险：只有 clean 数据，没有统一协议下的 DBLP 攻击文件。

应对：

- 在主实验前完成源文件清单；
- 记录攻击生成代码、参数和哈希；
- 不使用其他 split 或其他预算定义下的结果代替。

### 11.4 正式结果与开发结果混淆

风险：CPU smoke、golden、调试运行和正式 GPU 运行进入同一 protocol。

应对：

- 使用不同 protocol 名称；
- 正式结果只接受指定 GPU 环境和 commit；
- 汇总前过滤 protocol；
- 对 outputs 做只读归档。

### 11.5 长矩阵运行中断

风险：220+140 次运行中出现单次失败、机器重启或磁盘不足。

应对：

- 使用 Runner 自动跳过已完成实验；
- 开启 `--continue-on-error`；
- 定期检查失败 `status.json`；
- 分数据集和 suite 执行；
- 提前估算磁盘、显存和运行时间。

## 12. 最终完成定义

项目达到可发布状态需要同时满足：

- 阶段 1–3 工程代码稳定；
- Python 3.11 GPU 环境可复现；
- ACM 和 DBLP 全部 artifact 完整；
- HSeCo 和 DVCL golden 覆盖通过；
- 220 次主实验完成；
- 140 次消融实验完成；
- 汇总表和 Attack Average 完整；
- 所有结果均可追溯到配置、Git commit、环境和输入哈希；
- README、模型说明、开发路线和复现命令保持同步；
- 原始 outputs 和环境快照完成独立归档。
