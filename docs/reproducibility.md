# 最终复现与冻结说明

## 1. 冻结范围

最终冻结清单覆盖：

- Git 提交和工作树状态；
- Python、Torch、CUDA、DGL、PyG、GPU、驱动和全部 Python 包版本；
- 13 套论文协议、4124 次物理运行的完整性审计；
- clean、split、PRBCD、HetePRBCD、RND、HG 和自适应请求 artifact 哈希；
- 各协议 `metrics.json` 与 `manifest.json` 的集合哈希；
- 最终结果文档与 PNG/PDF 图表哈希。

配置入口为 `configs/reproducibility/final_freeze.yaml`。正式冻结默认拒绝脏工作树。

## 2. 最终冻结命令

先提交最终代码和文档，再执行：

```bash
source scripts/activate_gpu_env.sh
pytest -q
python scripts/check_environment.py \
  --profile gpu --smoke \
  --output outputs/environment/final-gpu-validation.json
python scripts/check_contracts.py
python scripts/analyze_paper_results.py
python scripts/freeze_reproducibility.py
```

验收条件：

- `outputs/reproducibility/final_manifest.json` 中 `publication_ready=true`；
- 13 套协议全部 `completed=expected`，合计 4124/4124；正式自适应矩阵另展开为 297/297 个预算结果；
- `git.dirty=false`；
- `cuda_available=true`；
- `git diff --check` 和完整 PyTest 通过。

开发期间可使用 `--allow-dirty` 生成预检清单，但该清单不能作为论文发布锁。

## 3. 正式实验复现命令

```bash
source scripts/activate_gpu_env.sh

python scripts/run_suite.py \
  --config configs/protocols/aminer_poisoning_main_v1.yaml \
  --continue-on-error
python scripts/run_suite.py \
  --config configs/protocols/aminer_rnd_poisoning_v1.yaml \
  --continue-on-error
python scripts/run_suite.py \
  --config configs/protocols/aminer_hg_baseline_target_evasion_v1.yaml \
  --continue-on-error

python scripts/run_suite.py \
  --config configs/protocols/acm_dblp_attack_seed_recheck_v1.yaml \
  --continue-on-error
python scripts/run_suite.py \
  --config configs/protocols/dvcl_adaptive_target_evasion_v1.yaml \
  --continue-on-error

python scripts/run_adaptive_pool.py \
  --config configs/protocols/adaptive_target_evasion_v1.yaml \
  --device cuda:0 --device cuda:1 --device cuda:2 \
  --continue-on-error --oom-retries 12 --oom-retry-delay 300
python scripts/analyze_adaptive_pilot.py \
  --config configs/protocols/adaptive_target_evasion_v1.yaml \
  --output-root outputs/analysis/adaptive_target_evasion_v1
```

Runner 自动跳过已完成运行；需要覆盖时显式添加 `--force`。

## 4. 结果与图表再生成

```bash
python scripts/analyze_paper_results.py
```

该命令生成：

- `outputs/paper_analysis/*.csv`：逐条件汇总、下降幅度、显著性、平均排名、实际预算利用率、正式自适应矩阵和 AMiner 攻击审计；
- `docs/final-experiment-results.md`：最终统计章节；
- `docs/aminer-experiment-results.md`：AMiner 11 模型结果；
- `docs/target-evasion-results.md`：迁移与自适应目标逃逸结果；
- `docs/figures/paper/`：PNG/PDF 论文图。

所有表格只报告 Micro-F1。Poisoning、迁移逃逸和自适应逃逸分别统计，不混合平均。
