# Routing-Level Signatures of Cognitive Efficiency

**A Multi-Task fMRI Null with a Pairwise-Correlation Caveat**

投稿：**IEEE BIBM 2026** · 会议版 · 在投

---

## 论点

高效认知是否对应某个可测量的皮层信息路由签名？具体地说，它是否 **(i) 更稀疏**、**(ii) 更符合任务典型的网络选择**、**(iii) 跨被试更一致**、还是 **(iv) 更模块化**？

本文在 101 名 HCP 被试、7 个任务 fMRI 范式上联合检验这四个预先设定的候选属性。结论是 **null**：

> 经 BH-FDR 校正后，**没有任何一个候选路由属性**能作为认知效率的签名通过显著性检验。

真正值得报告的发现是一个**统计陷阱**：在同一批路由一致性数据上，传统的 pairwise-correlation *t* 检验与 independence-safe 的被试级统计量之间，*p* 值相差 **16 个数量级**。这是个可复现的推断错误，对个体差异类功能连接研究有普遍警示意义。

## 数据

HCP Young Adult，101 个被试，7 个任务 fMRI 范式。数据在 `data/`（符号链接到共享的 `../../hcp_data/`，581 GB，只读）。

- HCP-MMP1.0 Glasser-360 分区（Cole-Anticevic CIFTI labels）
- Cole-Anticevic 12 网络划分，收敛为 7 个 Yeo-like 网络（VIS, SMN, DAN, VAN, LIM, FPN, DMN）
- 各范式的任务典型 GLM 对比（WM: 2-back vs 0-back；LANGUAGE: story vs math；SOCIAL: mental vs rnd；EMOTION: fear vs neut 等）

本文的**第三项贡献**是跨域对比：三个 MoE 基础模型（Switch-base-8 / Qwen1.5-MoE-A2.7B /
DeepSeek-MoE-16B）的专家路由 vs 大脑皮层，发现大脑的聚合稀疏度约为 MoE 的 **2.5 倍**
（正文 Table II）。模型权重在 `models/`（符号链接到 `../../moe_models`，与清华学报那篇共享只读）。

## 分析管线

```bash
python run_full_pipeline.py           # 全部 7 个 stage
python run_full_pipeline.py --list    # 查看 stage 列表
```

| Stage | 脚本 | 内容 |
|---|---|---|
| 1 | `s01_multi_task_extraction.py` | 从 7 个 HCP 任务提取激活模式 |
| 2 | `s02_sparsity_analysis.py` | H1：激活稀疏性 |
| 3 | `s03_routing_patterns.py` | H2/H3：任务-网络路由模式 |
| 4 | `s03b_h3_robust.py` | **H3 的 independence-safe 重检验**（LOO-r + label permutation）—— 16 个数量级差距就出自这里 |
| 5 | `s04_functional_modularity.py` | H4：Louvain 社区检测与模块化 |
| 6 | `s06b_hypothesis_fdr.py` | **跨 H1/H3/H4 的全局 BH-FDR 校正** |
| 7 | `s07_advanced_visualization.py` | 出版级图表 |

## 自动重跑

`tools/auto_rerun.sh` 会重跑脑侧分析、用 `tools/update_manuscript_numbers.py` 更新 `papers/conference_BIBM2026/BIBM2026_paper.tex` 里的正文数字，并重新编译。

## 稿件

`papers/conference_BIBM2026/`

- `BIBM2026_paper.tex` / `.pdf` — 当前版本
- `BIBM2026_paper.tex.bak` — 论点转向前的早期版本（标题为 *Inter-Subject Routing Consistency, Not Activation Sparsity, Distinguishes…*），差 193 行，保留作为思路演进记录
- `REPOSITION_DRAFT.md` — 改版思路
- `papers/figures_source/figures.pptx` — 正文 `Picture*.pdf` 的示意图源文件（承自最初被拒的 CogSci 版）

## 沿革

最初有一个 CogSci 2026 版本，主张存在一个**正向**的"高效大脑路由签名"。该版被拒后，同一批数据经 BH-FDR 校正重新审视，结论反转，才有了现在这篇 null-result 报告。CogSci 版稿件已删除。

## 与另一个项目的关系

同一批 HCP 数据还支撑着另一篇独立的论文——`../../TST_Journal_2026/Routing_MoE_TST/`（清华学报 TST，
大脑 vs MoE 基础模型的跨域路由对比）。两个项目**代码、结果、稿件完全独立**，只共享只读的 `../../hcp_data/` 与 `../../moe_models/`。
