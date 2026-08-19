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

截至 2026-08-04，`outputs/runs/model_golden/` 和
`outputs/equivalence/model_golden/` 下没有运行或比较报告，因此本轮完成的是
HAN/HeteroSAGE 基线实验，不是 13 条 Golden 对照。当前工作区仍为 dirty，严格
Golden 默认会拒绝启动；提交后需按下述命令执行。

## 执行

提交当前代码后，在已激活的 GPU 环境中执行：

```bash
source scripts/activate_gpu_env.sh
python scripts/run_golden_suite.py --reference-root ../HSeCo --dry-run
python scripts/run_golden_suite.py --reference-root ../HSeCo --continue-on-error
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
