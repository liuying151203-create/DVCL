# HSeCo 基线说明

HSeCo 对应 Zhao 等发表于 CIKM 2025 的《Robust Heterogeneous GNNs via Semantic Attention and Contrastive Learning》，DOI：`10.1145/3746252.3761343`。

## 方法与实验

HSeCo 由语义感知边权净化和结构对比优化组成，训练目标为分类交叉熵与对比损失之和。论文实验设置包括：

- 数据集：DBLP、ACM、AMiner；
- 元路径：DBLP 使用 APA、APTPA、APCPA，ACM 使用 PAP、PSP，AMiner 使用 PAP、PRP；
- 中毒攻击：PRBCD、HetePRBCD，扰动率为 0%、5%、10%、15%、20%、25%；
- 逃逸攻击：HG Baseline，预算为 0、1、3、5；
- 指标：Micro-F1；
- 基线：GCNJaccard、EGCNGuard、RGCN、SimPGCN、RoHe、HeteroSAGE、HeteroGuard、FastRoHGCN；
- 消融：HSeCo-c 移除语义感知边权模块，HSeCo-s 移除结构对比模块。

## 等效实现方式

本项目可综合参考论文与可获得源码。源码中完整的模型、预处理或训练逻辑可以经过接口适配后借鉴；缺失部分按照论文公式和实验行为补全。最终实现统一接入 DVCL Bench 的固定数据、划分、攻击和评估流程，而不是直接沿用来源代码各自的实验协议。

论文没有明确给出相似度函数、部分阈值、扰动视图方式、完整超参数、数据划分及早停规则。这些项目必须作为配置或复现假设记录，并在验证集上按与其他基线相同的预算选择。

`src/dvcl_bench/models/hseco.py` 已包含模型核心，数据与训练适配仍需完成。正式进入主表前，应通过张量与损失测试、消融行为检查以及多随机种子实验。结果应标记为“HSeCo 等效复现”，与论文报告数字分开保存。
