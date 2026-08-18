# 扩展实验实施计划

## 1. 当前状态

ACM 与 DBLP 的 HAN、HeteroSAGE、HSeCo、DVCL 全局 poisoning 主实验已经完成。
本轮新增的工程能力如下：

- 原生 RoHe、HeteroGuard 与 FastRoHGCN；
- HG Baseline 逐目标 evasion artifact、验证和七个模型 Trainer 的 clean-train/attack-eval 分支；
- ACM 500 个、DBLP 372 个固定目标节点，以及 \(\Delta\in\{1,3,5\}\) 的 6 个攻击 artifact；
- ACM/DBLP 的 RND 5%–25% 共 10 个 artifact；
- PRBCD/HetePRBCD surrogate 前后指标和优化曲线生成入口；
- `constrained` 与 `biased` 因子对照协议；
- AMiner 6564-paper 协议的严格数据 loader；
- suite 结果完整性审计和攻击有效性审计。

新增代码已经通过完整测试和三个新增基线的 ACM clean 单 epoch Runner smoke。
正式扩展实验尚未运行，不能把 smoke 指标写入论文表。

初版设计还计划通过 OpenHGNN 接入 HGT、MAGNN、HeCo 和 SimpleHGN。当前仓库只保留
模型名称映射，`run_experiment.py` 仍会阻断 `backend=openhgnn`，且正式环境尚未安装
OpenHGNN。因此这四个通用异构图基线尚未完成，不能计入下述七模型矩阵。HGSL 仅存在
于早期适配白名单，没有进入初版 README 的正式基线清单，是否纳入应单独立项。

## 2. 协议矩阵

| 优先级 | 配置 | 内容 | 运行数 |
|---:|---|---|---:|
| 1 | `robust_baselines_poisoning_v1.yaml` | RoHe、HeteroGuard、FastRoHGCN；ACM/DBLP；Clean、PRBCD、HetePRBCD | 330 |
| 2 | `hg_baseline_target_evasion_v1.yaml` | 7 个模型；ACM/DBLP；\(\Delta=1,3,5\)；target-only Micro-F1 | 210 |
| 3 | `rnd_poisoning_v1.yaml` | 7 个模型；ACM/DBLP；RND 5%–25% | 350 |
| 4 | `attack_factorial_v1.yaml` | PRBCD unconstrained、HetePRBCD w/o biased；5/15/25% | 240 |
| 数据就绪后 | `aminer_poisoning_main_v1.yaml` | 7 个模型；Clean、PRBCD、HetePRBCD | 385 |
| 数据就绪后 | `aminer_hg_baseline_target_evasion_v1.yaml` | 7 个模型；$\Delta=1,3,5$ | 105 |
| 数据就绪后 | `aminer_rnd_poisoning_v1.yaml` | 7 个模型；RND 5%–25% | 175 |
| 待接入 | OpenHGNN 通用基线 suite | HGT、MAGNN、HeCo、SimpleHGN | 待冻结协议 |

所有正式条件使用 split seed 1、attack seed 1 和 train seed 1–5。目标逃逸与全局
poisoning 分别汇总，禁止计算到同一个 Attack Average。

## 3. 攻击诊断

正式 PRBCD 使用 `constrained=true, biased=false`；正式 HetePRBCD 使用
`constrained=true, biased=true`。因子对照只改变一个因素：

- `prbcd_unconstrained`：检验语义约束对攻击可实现性和破坏性的影响；
- `heteprbcd_unbiased`：检验 biased sampling 对攻击强度的贡献；
- RND：提供同扰动率的非优化随机结构扰动对照。

现有正式 PRBCD/HetePRBCD artifact 没有保存 surrogate 前后 Micro-F1 和优化历史。
`generate_prbcd_diagnostic_source.py` 会在新 source 中保存
`surrogate_before`、`surrogate_after` 和 `optimization_history`；
`analyze_attack_effectiveness.py --strict-generation-diagnostics` 会阻断缺失这些字段的审计。

`adaptive=true` 只允许 surrogate 与被评估 victim 的模型族和参数真正一致。当前外部生成器
的 PRBCD surrogate 是 GCN，HetePRBCD surrogate 是 HeteroSAGE，不能把其产物改名为
HAN/HSeCo/DVCL 自适应攻击。HSeCo/DVCL 含 SciPy 阈值净化和离散构图，后续需要 BPDA
或可微替代算子；在该研究实现完成前，自适应结果不进入论文主表。

## 4. AMiner

工程已支持 `data/aminer/pa.npz`、`pr.npz`、`labels.npy`、`pos.npz`，并严格要求：

- 6564 个 paper；
- Paper–Author 与 Paper–Research 两类双向关系；
- 4 类 paper 标签；
- 20%/10%/70% 的 paper protocol split。

本机只有 AMiner split 和 HG Baseline 攻击记录，没有上述四个原始图文件。PyG 的 AMiner
节点/关系定义不同，不能替代。拿到同版原始文件后依次执行 clean、split、HG Baseline、
PRBCD/HetePRBCD 和七模型实验，并在导入时核对旧攻击目标与新 clean 图。

```bash
python scripts/prepare_dataset.py --dataset aminer
python scripts/generate_split.py --dataset aminer --seed 1 --protocol paper
python scripts/prepare_hg_baseline_artifacts.py \
  --dataset aminer \
  --source-root PATH_TO_HG_BASELINE_SOURCES

python scripts/run_suite.py --config configs/protocols/aminer_poisoning_main_v1.yaml --continue-on-error
python scripts/run_suite.py --config configs/protocols/aminer_hg_baseline_target_evasion_v1.yaml --continue-on-error
python scripts/run_suite.py --config configs/protocols/aminer_rnd_poisoning_v1.yaml --continue-on-error
```

其中 PRBCD/HetePRBCD 源 artifact 必须来自同一 clean 图；在它们导入正式路径前，
`aminer_poisoning_main_v1` 只能 dry-run，不能启动正式训练。

## 5. 执行与审计

```bash
source scripts/activate_gpu_env.sh

python scripts/run_suite.py --config configs/protocols/robust_baselines_poisoning_v1.yaml --continue-on-error
python scripts/run_suite.py --config configs/protocols/hg_baseline_target_evasion_v1.yaml --continue-on-error
python scripts/run_suite.py --config configs/protocols/rnd_poisoning_v1.yaml --continue-on-error

python scripts/prepare_attack_factorial.py
python scripts/run_suite.py --config configs/protocols/attack_factorial_v1.yaml --continue-on-error
```

每个 suite 完成后立即执行：

```bash
python scripts/audit_suite_results.py --config CONFIG_PATH
python scripts/summarize_results.py
```

PRBCD 诊断执行：

```bash
python scripts/analyze_attack_effectiveness.py \
  --protocol acm_poisoning_main_v1 \
  --protocol dblp_poisoning_main_v1 \
  --output outputs/audits/attack_effectiveness/current.csv
```

论文表只采用完整 suite、干净 Git 提交、通过 manifest/artifact 审计的 Micro-F1 均值与
样本标准差。新增结果依次写入鲁棒基线表、目标逃逸表、RND 对照表和攻击因子分析表。
