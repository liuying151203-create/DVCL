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
| 当前状态 | 攻击 artifact 已验证，正式训练待运行 |

### 攻击实现版本

| 仓库 | Commit |
|---|---|
| DVCL 实验编排 | `da3cc8b` |
| HSeCo | `1ebab2c` |
| Hetero-Guard | `b6375a0` |

正式 artifact 的 `promotion.json` 记录代码提交、clean/split/attack 哈希和预算验证。
旧攻击产物归档在 `outputs/archive/attacks_pre_protocol_v2`，不再用于正式训练。

### 运行结果

正式实验完成后在此记录完整性、Micro-F1 表格、异常运行及处理结论。
