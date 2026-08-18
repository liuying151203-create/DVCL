# 实验运行记录

## DBLP 修正投毒协议

| 项目 | 内容 |
|---|---|
| Protocol | `dblp_poisoning_main_v1` |
| 数据集 | DBLP |
| 模型 | HAN、HeteroSAGE、HSeCo、DVCL |
| 攻击 | Clean、PRBCD、HetePRBCD |
| 扰动率 | 5%、10%、15%、20%、25% |
| Seeds | split 1，attack 1，train 1–5 |
| 训练 | 200 epochs，patience 100 |
| 预期运行数 | 220 |
| 结果指标 | Micro-F1 |
| 当前状态 | 220/220 完成，已汇总并写入正式 DBLP 结果表 |

### 攻击实现版本

| 仓库 | Commit |
|---|---|
| DVCL 实验编排 | `da3cc8b` |
| DVCL 分片 Runner | `1175fe7` |
| HSeCo | `1ebab2c` |
| Hetero-Guard | `b6375a0` |

正式 artifact 的 `promotion.json` 记录代码提交、clean/split/attack 哈希和预算验证。
旧攻击产物归档在 `outputs/archive/attacks_pre_protocol_v2`，不再用于正式训练。

### 执行方式

正式矩阵按模型分成 4 个互不重叠的分片，每个分片包含 55 次顺序运行，分别使用
`cuda:0` 至 `cuda:3`。各分片日志保存在
`outputs/logs/dblp_poisoning_main_v1`。

### 运行结果

- 完整性：220 个 `status.json` 均为 `completed`，220 个 `metrics.json` 完整。
- 审计：220 个 manifest 均为 `git_dirty=false`；clean/split 哈希分别唯一，攻击哈希
  共 10 个，与正式 artifact 一一对应。
- 汇总：`outputs/summaries/dblp_poisoning_main_v1`。
- 正式结果：`docs/dblp-experiment-results.md`。
- 分片退出码：HAN、HeteroSAGE、HSeCo、DVCL 均为 0，无失败重跑。
- 异常观察：HAN 在 HetePRBCD 25% 下方差较大；HetePRBCD 各扰动率对训练节点
  高度集中，均已在结果分析中披露。

## ACM 修正投毒协议

| 项目 | 主实验 | 消融实验 |
|---|---|---|
| Protocol | `acm_poisoning_main_v1` | `acm_poisoning_ablation_v1` |
| 配置 | `configs/protocols/acm_poisoning_main_v1.yaml` | `configs/suites/acm_poisoning_ablation_v1.yaml` |
| 模型 | HAN、HeteroSAGE、HSeCo、DVCL | DVCL 四个组件 variant |
| 攻击 | Clean、PRBCD、HetePRBCD | Clean、PRBCD、HetePRBCD |
| 扰动率 | 5%、10%、15%、20%、25% | 5%、15%、25% |
| Seeds | split 1，attack 1，train 1–5 | split 1，attack 1，train 1–5 |
| 训练 | 200 epochs，patience 100 | 200 epochs，patience 100 |
| 预期运行数 | 220 | 140 |
| 结果指标 | Micro-F1 | Micro-F1 |
| 输入审计 | 12/12 通过 | 8/8 通过 |
| 当前状态 | 220/220 完成 | 140/140 完成 |

### 攻击产物审计

修正 artifact 已提升至 `data/attacks/acm`，旧产物归档至
`outputs/archive/attacks_pre_protocol_v2/acm`。所有条件均严格用满全局预算；训练集
变化占比用于披露攻击分布，不作为通过条件。

| Attack | Rate | Expected | Actual | Train Change Share |
|---|---:|---:|---:|---:|
| PRBCD | 5% | 871 | 871 | 91.62% |
| PRBCD | 10% | 1743 | 1743 | 89.62% |
| PRBCD | 15% | 2614 | 2614 | 87.95% |
| PRBCD | 20% | 3486 | 3486 | 85.74% |
| PRBCD | 25% | 4358 | 4358 | 83.23% |
| HetePRBCD | 5% | 871 | 871 | 98.62% |
| HetePRBCD | 10% | 1743 | 1743 | 96.62% |
| HetePRBCD | 15% | 2614 | 2614 | 94.22% |
| HetePRBCD | 20% | 3486 | 3486 | 92.03% |
| HetePRBCD | 25% | 4358 | 4358 | 89.88% |

`promotion.json` 记录 DVCL `8a354c4`、HSeCo `1ebab2c`、Hetero-Guard
`b6375a0` 的干净提交以及 clean、split、attack 哈希。

### 执行分片

- 主实验按模型分为 4 个逻辑分片，每片 55 次运行。HSeCo 为缩短尾部时间，再按
  PRBCD 和 HetePRBCD 扰动率拆成互不重叠的执行分片；最终使用 `cuda:1`–`cuda:4`。
- 消融按 `full`、`no_cl`、`topology_only`、`feature_only` 分为 4 个分片，
  每片 35 次运行；论文表中统一显示为 Full DVCL 或 `w/o ...`。
- 首次启动三个主实验分片时绕过了 `scripts/activate_gpu_env.sh`，DGL 因缺少
  `libcusparse.so.12` 动态库路径而立即失败。停止分片并恢复完整激活流程后，所有
  失败状态均在原 run_dir 中被成功运行覆盖；最终汇总不包含这次启动失败的指标。

### 运行结果

- 完整性：主实验 220/220、消融 140/140，所有 `status.json` 均为 `completed`。
- 审计：360 个 manifest 均为 `git_dirty=false`，代码提交均为 `bf8c181`。
- 主实验汇总：`outputs/summaries/acm_poisoning_main_v1`。
- 消融汇总：`outputs/summaries/acm_poisoning_ablation_v1`。
- 正式结果：`docs/acm-experiment-results.md`。
- 主实验四模型与消融四 variant 的最终分片退出码均为 0，无残留失败状态。
