# Golden 对照

Golden 对照用于验证相邻旧 HSeCo 仓库与当前原生实现是否在相同代码环境、
冻结 artifact、随机种子和设备下产生一致训练轨迹。它不是论文数字与当前结果的
人工比较。

## 对照矩阵

`configs/golden/hseco_dvcl.yaml` 定义 13 个 train seed 1 条件：

- HSeCo：ACM clean、PRBCD 5%/25%、HetePRBCD 5%/25%，以及 DBLP clean、
  PRBCD 15%、HetePRBCD 15%；
- DVCL：ACM clean、PRBCD 5%、HetePRBCD 5%，以及 DBLP clean、
  HetePRBCD 15%。

每个条件的 reference 和 current 读取完全相同的 clean、split 和 attack 文件。
配置强制 `legacy_checkpoint_semantics=true`，并默认要求两个仓库均为干净状态。
参考侧由 `scripts/run_legacy_reference.py` 仅将原生字典 artifact 解码为旧入口所需的
DGL 图和张量，模型、损失、优化、早停及指标计算仍执行相邻 HSeCo 仓库代码。

## 当前状态

截至 2026-08-20，13 条 GPU Golden 对照全部通过，容差为 0。参考仓库提交为
`1ebab2c`，当前仓库提交为 `85ede0d`；两仓库在运行时均为 clean。各条件的验证
Micro-F1 轨迹最大差异和最终 Micro-F1 差异均为 0。

| Model | Dataset | Condition | Epochs compared | Result |
|---|---|---|---:|---|
| HSeCo | ACM | Clean | 111 | Pass |
| HSeCo | ACM | PRBCD 5% / 25% | 200 / 187 | Pass |
| HSeCo | ACM | HetePRBCD 5% / 25% | 118 / 150 | Pass |
| HSeCo | DBLP | Clean | 200 | Pass |
| HSeCo | DBLP | PRBCD 15% / HetePRBCD 15% | 114 / 110 | Pass |
| DVCL | ACM | Clean | 118 | Pass |
| DVCL | ACM | PRBCD 5% / HetePRBCD 5% | 120 / 120 | Pass |
| DVCL | DBLP | Clean / HetePRBCD 15% | 120 / 200 | Pass |

## 执行

提交当前代码后，在已激活的 GPU 环境中执行：

```bash
source scripts/activate_gpu_env.sh
python scripts/run_golden_suite.py --reference-root ../HSeCo --dry-run
python scripts/run_golden_suite.py --reference-root ../HSeCo --continue-on-error --force
```

`DVCL_PRIVATE_HSECO_ROOT` 可替代 `--reference-root`。仅调试时才允许使用
`--allow-dirty`，其结果不能作为严格 Golden。

## 输出与判定

输出位于 `outputs/equivalence/model_golden/`，每个条件包含：

- `audit.json`：两仓库提交、dirty 状态、输入哈希和完整命令；
- `reference_stdout.log` 与 `reference_metrics.log`；
- `report.json`：逐 epoch 验证轨迹和最终指标差异；
- 顶层 `summary.json`：全部条件的汇总状态。

当前配置要求同环境严格零差异，即 `tolerance=0`。DVCL 的 reference 是相邻旧
HSeCo 仓库中的 DVCL 实现，用于迁移回归审计，不表示存在独立发表的 DVCL
官方实现。
