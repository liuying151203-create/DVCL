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
