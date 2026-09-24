# Routing-Level Signatures of Cognitive Efficiency

**Only Inter-Subject Routing Consistency Survives FDR, with a Pairwise-Correlation Caveat**

投稿：**IEEE BIBM 2026**（regular paper B325）· 已录用 · camera-ready 在 `camera-ready` 分支

---

## 论点

高效认知是否对应某个可测量的皮层信息路由签名？具体地说，它是否 **(i) 更稀疏**、**(ii) 更符合任务典型的网络选择**、**(iii) 跨被试更一致**、还是 **(iv) 更模块化**？

本文在 100 名 HCP 被试、7 个任务 fMRI 范式上联合检验这四个预先设定的候选属性。camera-ready 版的结论是：经 BH-FDR 校正后，**只有被试间路由一致性（H3）通过**（被试级 LOO Welch *d* = +0.52，*p* = 0.011，*p*<sub>FDR</sub> = 0.033；置换 *p* = 0.018；与连续效率分数 *r* = 0.36），稀疏度、典型网络选择和模块度都不支持。

方法学上的警示仍然成立：同一批一致性数据，传统的 pairwise-correlation *t* 检验给出 *p* = 1.0×10<sup>−30</sup>，比被试级统计量夸大约 28 个数量级。

投稿版曾报告“四项全为 null”，那是在一个有误的一级 GLM 上得到的（见下文“camera-ready 修正”）。

## 数据

HCP Young Adult，100 个被试（一份按 ID 升序排列、7 个任务 LR/RL 齐全的名单中的前 100 人），7 个任务 fMRI 范式。原始数据原先通过 `data -> ../../hcp_data` 链接到共享目录，该目录已于 2026-09-24 删除；现在 `data/` 只保留 `_atlas/` 图谱文件（Glasser-360、Cole-Anticevic、MNI 体积图谱），重跑一级 GLM 前需要先用 `tools/download_hcp.py` 重新下载 HCP 数据。组级与被试级中间结果都在 `results/`，不依赖原始数据。使用的是 MSMAll 最小预处理数据（未做 ICA-FIX），入组时没有做头动筛选（事后核查：所有 run 的平均相对位移都小于 0.5 mm）。

- HCP-MMP1.0 Glasser-360 分区（Cole-Anticevic CIFTI labels）
- Cole-Anticevic 12 网络划分，收敛为 7 个 Yeo-like 网络（VIS, SMN, DAN, VAN, LIM, FPN, DMN）
- 任务对比按 HCP 标准：WM 2-back vs 0-back；MOTOR 五种运动 vs 固视；LANGUAGE story vs math；SOCIAL mental vs rnd；RELATIONAL relation vs match；EMOTION fear faces vs shapes；GAMBLING win vs loss

跨域对比用三个 MoE 模型（Switch-base-8 / Qwen1.5-MoE-A2.7B / DeepSeek-MoE-16B）。MoE 路由数据取自清华学报姊妹项目修正过 router hook 之后的结果（`../../TST_Journal_2026/Routing_MoE_TST/results/run_20260419_182908/moe_analysis/`）。按层合并时皮层比 MoE 集中约 3.3 倍，但 MoE 单层可以比皮层更集中。

## camera-ready 修正（2026-09-24）

- **一级 GLM**（`scripts/s01b_glm_fixed.py`）：原 `s01` 把 EVs 目录下所有 `.txt` 都当回归量（包括 Sync 和与组块重叠的事件 EV，WM 的设计矩阵秩亏），按子串匹配 EV 名，HRF 下冲项分母写错，MOTOR 对比 cue。修正版只用组块 EV、精确匹配、SPM HRF、128 s 高通，MOTOR 对固视。结果在 `results/run_20260924_glmfix/`；原 `results/run_20260419_182908/` 保持不动。
- **H1** 改为被试级检验（原来把 700 个被试×任务值当独立观测）。
- **MoE** 数字改用修正后的 router 抽取；原 H7 的 r=+0.25（截断展平矩阵逐元素相关）改为 Mantel 检验。
- **审稿人要求的补充分析**：`scripts/s08_camera_ready_analyses.py`；新图：`scripts/s09_camera_ready_figures.py`；一键重跑：`scripts/run_glmfix.sh`。
- 投稿用源文件包：`papers/conference_BIBM2026/B325_camera_ready_source.tar.gz`（未纳入 git）。

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
大脑 vs MoE 基础模型的跨域路由对比）。两个项目**代码、结果、稿件完全独立**，原先共享 `../../hcp_data/` 与 `../../moe_models/`，两者已于 2026-09-24 删除。
