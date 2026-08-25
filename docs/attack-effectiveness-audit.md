# 攻击有效性审计

> **专项审计。** 论文主结果及其结论边界以 `docs/final-experiment-results.md` 为准。

## 审计范围

- 协议：ACM/DBLP 主模型、鲁棒基线和 OpenHGNN 基线 poisoning，共 1,210 次运行。
- 条件：PRBCD、HetePRBCD，$\rho\in\{5,10,15,20,25\}\%$。
- 审计表：`outputs/audits/attack_effectiveness/current.csv`，共 220 个模型条件。
- 检查项：实际预算、关系变化、训练集变化占比、训练集富集倍数及 Micro-F1 降幅。

## 结果

| Dataset | Attack | Train-change share | Enrichment | Micro-F1 drop range |
|---|---|---:|---:|---:|
| ACM | PRBCD | 83.23%–91.62% | 4.16–4.58× | -0.43–10.02 |
| ACM | HetePRBCD | 89.88%–98.62% | 4.49–4.92× | -0.48–11.49 |
| DBLP | PRBCD | 53.56%–97.45% | 5.37–9.76× | -1.92–44.37 |
| DBLP | HetePRBCD | 93.58%–99.60% | 9.37–9.98× | 1.38–69.10 |

## 结论

1. 所有正式 artifact 的实际预算与配置一致，实验下降不是预算缺失导致。
2. DBLP HetePRBCD 几乎将全部变化集中到训练节点邻域，解释了其显著强于 PRBCD
   和 RND 的破坏性；攻击因子实验进一步确认 biased sampling 是主要来源。
3. DBLP HetePRBCD 下 DVCL 的平均 Micro-F1 为 82.40，高于 HSeCo 的 75.88；
   但该结论属于固定全局 poisoning，不代表针对 DVCL 优化的自适应攻击鲁棒性。
4. 旧 PRBCD/HetePRBCD artifact 不含优化轨迹和 surrogate 前后指标，因此生成过程只能
   审计预算与结构变化；新攻击因子 artifact 已补齐这些诊断并通过严格审计。
