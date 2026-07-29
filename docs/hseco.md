# HSeCo 基线说明

HSeCo 对应 CIKM 2025 论文《Robust Heterogeneous GNNs via Semantic Attention and Contrastive Learning》，DOI：`10.1145/3746252.3761343`。

论文使用 DBLP、ACM、AMiner，包含 PRBCD、HetePRBCD 全局攻击和 HG Baseline 目标攻击。主要对比方法包括 RoHe、HeteroSAGE、HeteroGuard 和 FastRoHGCN。

本项目采用等效实现：

- 数据与划分通过版本化 artifact 固定；
- 按 HSeCo 源码流程计算元路径转移权重和两级语义净化；
- 保留语义 HAN、节点聚合器、分类损失与结构对比损失；
- 保留旧工程的 loss/accuracy 联合早停规则；
- 支持旧式 checkpoint 行为，用于与原工程做复现审计；
- 统一主协议保存并恢复语义模块和分类模块，再重新生成语义图。

当前原生实现位于：

- `src/dvcl_bench/models/semantic.py`
- `src/dvcl_bench/adapters.py`
- `configs/models/hseco_native.yaml`

结果应标记为“HSeCo 等效复现”，并与论文报告数字分开保存。正式主表只使用与 DVCL 共用数据、划分、攻击和调参预算后重新运行的结果。
