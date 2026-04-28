# 国奖目标版正文扩充与图表回填报告

**生成时间**：2026-04-28
**任务**：在严格符合附件 1 格式规范的前提下，将 `main.tex` 正文从约 13 页扩充到 18–19 页饱满状态，回填高价值图表，强化关键解释。

---

## 1. 页数变化

| 指标 | 修改前 | 修改后 | 目标 |
|---|---:|---:|---|
| 12.3 结束页码 | p13 | **p18** | 18–19 |
| 是否达到 18–19 页目标 | 否（偏短） | **是** | — |
| 是否低于 p20 硬上限 | 是 | **是** | ≤ 20 |
| PDF 总页数（含参考文献+附录） | 23 | 26 | — |
| 正文图数量（不含附录） | 5 | **8** | 6–8 ✓ |
| 正文表数量（不含附录） | 14 | 14 | 13–15 ✓ |

§12.3 展望与参考文献第 1 段同处 p18，参考文献延续到 p19，附录 A 起始于 p20。

---

## 2. 从附录回填到正文的图

| # | 图（label） | 文件 | 插入位置 | 作用 |
|---|---|---|---|---|
| 1 | `fig:q2-gantt` | `fig_rebuild_q2_time_window_gantt.pdf` | §7.3 Q2 三层求解后 | 直观呈现客户 14 晚到 37 单位、99.96% penalty 的结构性来源 |
| 2 | `fig:q3-qubit-budget` | `fig_rebuild_q3_qubit_budget.pdf` | §8.1 比特预算表后 | 突出 2500 vs 289/289/256 vs 550 上限的硬规模差距 |
| 3 | `fig:q4-routes` | `fig_rebuild_q4_multivehicle_routes.pdf` | §9.3 K=7 路径表后 | 全文最关键章节的多车 7 色拓扑路线 |
| 4 | `fig:solver-compare` | `fig_rebuild_solver_compare.pdf` | §10.1 三层一致性表后 | symlog 同图展示四问 Python/SDK/CIM 跨 4 量级业务目标 |
| 5 | `fig:q1-sens-body` | `fig_q1_sens_bestcost_heatmap.pdf` | §10.2 Q4 K 灵敏度后 | Q1 罚系数 A×初温 T0 二维稳健性热图 |

附录中对应位置的重复定义已删除以避免 `Duplicate label` 警告。

---

## 3. 重新生成的图

本次未额外生成新图。回填的 5 张图均直接来自项目已有的 `figures/` 与 `figures_rebuild_candidates/02_python_result_figures/` 目录，均为真实数据驱动的 PDF 矢量图，未对附录中已有版本做画风改造（避免不必要风险）。

---

## 4. 仍保留在附录的图与原因

| 图（label） | 保留原因 |
|---|---|
| `fig:q1-sens-app` | 5×5 灵敏度热图全量版（含 feasibility + bestcost 两子图）信息量大，正文已用简化版 |
| `fig:q1-cim-h` | Q1 CIM 10-sample 能量分布，正文 §10 已用 solver-compare 综合呈现 |
| `fig:q2-cim-h` | Q2 CIM 哈密顿量演化曲线，正文 §7 已用 gantt 替代 |
| `fig:q2-abl` | Q2 C/D 编码消融柱状图，正文已有 tab:q2-enc 表格表达 |
| `fig:q3-route` | Q3 完整 50 客户路径图，节点过密放正文不够清晰 |
| `fig:q3-cim-h` | Q3 CIM 能量分布，正文 §10 已综合 |
| `fig:q4-gantt` | Q4 多车甘特图，7 车×50 客户排版过挤 |
| `fig:q4-violation` | Q4 违反分布柱状图，正文 §9.3 已文字化呈现 |
| `fig:q4-pareto` | Q4 K=7 vs K=8 Pareto 散点图，正文已用 tab:q4-Ksens 与 §9.4 文字解释 |
| `fig:q4-cim-h` | Q4 七车 CIM 哈密顿量，正文 §9.5 已用三层对比表呈现 |
| `fig:q4-qubits` | Q4 每车比特预算柱状图，tab:q4-budget 已清楚呈现 |

---

## 5. 正文文本被补强的位置

| 章节 | 补强内容 |
|---|---|
| §2 问题分析 | 新增``四问递进关系''段（业务可行解 vs 真机可提交性双正交目标） |
| §6.2 Q1 三层求解 | 新增``Q1 在四问体系中的定位''段（Held–Karp 角色、A=200 折中、H 与 J 关系） |
| §7.3 Q2 三层求解 | 强化结构性特征解释（最短路 vs 时间窗最优分离、单车结构性现象） |
| §8.1 Q3 比特预算 | 新增对比图后段落（4.5× 硬规模差距、subQUBO 局部优化器定位） |
| §8.3 Q3 三层求解 | 强化 seg2/seg3 差异机制（段尾衔接被打乱）、CIM 配额诚信表述 |
| §9.3 Q4 K=7 主方案 | 新增覆盖容量、运输与违反、综合目标三段定量解析（容量利用 64.5%） |
| §9.4 K=7 vs K=8 | 强化 Pareto 双方案权衡（车辆资源 vs 客户体验） |
| §9.5 Q4 三层对比 | 强化时间差诚信定位、block-diag 配额节约 vs 样本数 trade-off |
| §10.1 一致性 | 新增 Python/SDK/CIM 三层 gap 来源解释（CIM 仅 10 sample 统计冗余小） |
| §10.2 灵敏度 | 新增 Q1 A×T0 热图分析、综合稳健性结论段 |
| §11.1 模型优势 | 4 项扩展为更具体表述（题目契合、模型递进、比特预算、链路一致） |
| §11.2 模型不足 | 4 项扩展为机制级解释（最优性证明、SA 邻域冲突、配额限制、分阶段优化） |
| §11.3 改进推广 | 3 项从行级扩展到段落级，引入 ILP/CP-SAT、SA 邻域改良、联合编码迁移 |
| §12.1 逐题结论 | 4 段每段从 2 行扩展到 4 行，含路径表达式、机制解释、关键百分比 |
| §12.2 综合结论 | 4 项从行级扩展到段落级，强调链路完整性与方法迁移性 |
| §12.3 展望 | 3 项分``算法层 / 验证层 / 模型层''三视角重写 |

---

## 6. 引用与交叉引用修复

- 编译三遍 `xelatex` 后 `main.log` 中无 `LaTeX Warning: Reference ... undefined` / `Label(s) may have changed` 警告。
- 所有图、表、公式引用（`\ref{fig:*}`、`\ref{tab:*}`、`\eqref{eq:*}`）均正常解析。
- 所有 `\cite{*}` 文献编号从 [1] 至 [22] 连续无缺。
- 全文不存在 `[?]`、`图 ??`、`表 ??`、`式 ??`。

---

## 7. 编译信息

| 项 | 值 |
|---|---|
| 编译器 | XeLaTeX (TeX Live 2023+, ctex 宏包) |
| 编译次数 | 3 次（首次解析 cite，第二次解析 forward refs，第三次定稿） |
| 编译命令 | `xelatex -interaction=nonstopmode -halt-on-error main.tex` |
| 输出 PDF | `main.pdf`（26 页） |
| Errors | 0 |
| Undefined refs | 0 |
| Overfull hboxes | 5 处（最大 116pt，集中在附录 src/ 路径与表格首行，影响轻微） |
| Underfull hboxes | 1 处（badness 10000，位于稳定性表内一段） |

---

## 8. 附件 1 格式规范合规性

| 规范要求 | 状态 |
|---|---|
| 第 1 页仅含题目+摘要+关键词 | ✓ |
| 第 2 页起为正文 | ✓ |
| 摘要 ≤ 1 页 | ✓ |
| 无目录、无图目录、无表目录 | ✓ |
| 无页眉 | ✓ `\pagestyle{plain}` |
| 无任何身份信息 | ✓ |
| 题目三号黑体居中 | ✓ `{\zihao{3}\CJKfamily{hei}}` |
| 一级标题四号黑体居中 | ✓ CTEXsetup section |
| 二/三级小四黑体左对齐 | ✓ CTEXsetup subsection/subsubsection |
| 正文小四宋体 | ✓ `\renewcommand{\normalsize}{\zihao{-4}}` |
| 单倍行距 | ✓ `\linespread{1.0}` |
| 引用为 [n] 形式 | ✓ thebibliography 自动 |
| 12.3 结束 ≤ p20 | ✓ p18 |

---

## 9. 关键数值前后一致性核对

| 问题 | 关键值 | 摘要 | 正文 | 结论 |
|---|---|---|---|---|
| Q1 | J = 29 / 29 / 31, gap +6.9% | ✓ | ✓ | ✓ |
| Q2 | J = 84121, travel = 31, penalty = 84090 | ✓ | ✓ | ✓ |
| Q2 | 12/15 客户违反 | ✓ | ✓ | ✓ |
| Q3 | J = 4,941,906, 分段 17/17/16, 子 QUBO 289/289/256 | ✓ | ✓ | ✓ |
| Q3 | 45/50 客户违反, CIM seg2 高 7.9%, seg1 未获返回 | ✓ | ✓ | ✓ |
| Q4 | C=60, ∑d=271, K_min=5, K*=7 | ✓ | ✓ | ✓ |
| Q4 | J=7149, travel=109, penalty=40, 3/50 违反 | ✓ | ✓ | ✓ |
| Q4 | CIM J=7168, gap +0.27%, K=8 J=8124 | ✓ | ✓ | ✓ |
| Q4 | 七车 demand 35,42,47,41,35,39,32 = 271 | ✓ tab:q4-k7-routes 与正文段落复算 | — |
| 真机 | 12 次配额, ising ≤ 550, 8-bit | ✓ | ✓ | ✓ |

---

## 10. 仍需人工核对的项

1. **附录图路径**：5 张回填图的源文件（`fig_rebuild_*.pdf`）位于 `figures_rebuild_candidates/02_python_result_figures/`，已通过 `\graphicspath{}` 加入搜索路径；如未来路径变动需同步更新。
2. **Q2 调度详表**：`tables/tab_02_q2_schedule.tex` 现引入附录 §E，建议人工核对 15 行 `t_i, P_i` 与 JSON `q2_pure_python.json` 的字段是否完全一致。
3. **CIM 任务清单 tab:cim-tasks**：附录 §D 中任务编号 R-Q1-005、R-Q2-006、R-Q3-seg1∼3、R-Q4-V1∼V7、R-Q4-K8-bd 共 12 条，建议核对 `results/真机结果/00_四问真机对比汇总.md` 的 task 编号一致性。
4. **Overfull hbox**：5 处轻微超宽（最大 116pt）集中在附录代码路径表，不影响阅读，可考虑下次细修时改为 `\texttt{}` 内嵌 `\allowbreak` 或换行。

---

## 11. 全文被禁用语清查（核查通过）

以下词汇在 `main.tex` 中**未出现**（grep 验证）：

国奖级 / 冲奖 / 最强证据 / 铁律 / 我赢 / 打平 / 微输 / 破解 / 救回 / 丢失 / 无虚构 / Claude Code / AI / 彻底证明 / 量子优势 / 显著加速 / 全面占优。

替换性表达均使用克制语言：``近似最优''、``真机求解链路可行''、``在当前实现和参数设置下耗时较低''、``经局部修复后恢复为可行路径''、``未获得有效返回''、``略低于参考''、``优于参考''等。

---

## 12. 论文最终结构（含页码）

| 页码 | 内容 |
|---|---|
| p1 | 题目 + 摘要 + 关键词 |
| p2 | §1 问题重述 + §2 问题分析（开头） |
| p3 | §2 问题分析（含 tab:prob-analysis 与 fig:overview） + §3 模型假设 |
| p4 | §4 符号说明 + §5 数据说明与预处理 + §6 Q1 模型 |
| p5 | §6 Q1 三层求解结果（含 tab:q1-cmp + fig:q1-route + 灵敏度段） |
| p6 | §7 Q2 模型 + 编码取舍（含 tab:q2-enc） |
| p7 | §7 Q2 三层求解（含 tab:q2-cmp + fig:q2-route） |
| p8 | §7 Q2 时间窗甘特图（fig:q2-gantt）+ 结构性特征 |
| p9 | §8 Q3 规模挑战与比特预算（含 tab:q3-budget + fig:q3-qubit-budget） |
| p10 | §8 Q3 滚动窗口分解 subQUBO |
| p11 | §8 Q3 三层求解（含 tab:q3-cmp + 详细分析） |
| p12 | §9 Q4 多车模型 + 比特预算 |
| p13 | §9 Q4 K=7 主方案路径表 |
| p14 | §9 Q4 多车路径图（fig:q4-routes）+ K 灵敏度（tab:q4-Ksens + fig:q4-Ksens） |
| p15 | §9 Q4 三层对比表 + §10 检验与一致性 |
| p16 | §10 三层 solver-compare 图（fig:solver-compare）+ Q1 灵敏度热图 |
| p17 | §10 稳定性 + §11 模型评价（优势、不足） |
| p18 | §11 改进推广 + §12 结论（12.1, 12.2, 12.3）+ 参考文献开始 |
| p19 | 参考文献 |
| p20-26 | 附录 A∼F |

---

## 13. 主要交付文件

| 文件 | 说明 |
|---|---|
| `paper/Manuscript/main.tex` | 主稿 LaTeX 源（已扩充至 18 页正文） |
| `paper/Manuscript/main.pdf` | 编译后 PDF（26 页：18 页正文+1 页参考文献+7 页附录） |
| `paper/Manuscript/body_enhancement_report.md` | 本扩充报告 |
| `paper/Manuscript/final_compression_report.md` | 上一轮压缩报告（保留供对照） |
| `paper/Manuscript/backup_before_final_compress/` | 上一轮压缩前备份 |

---

## 14. 任务完成度自评

- [x] 第 1 页仅题目+摘要+关键词
- [x] 第 2 页开始正文
- [x] 无目录、无页眉、无身份信息
- [x] §12.3 结束页码 = **p18**（在 18–19 页理想区间内）
- [x] §12.3 结束 ≤ p20 硬上限
- [x] 5 张高价值图回填正文
- [x] Q1/Q2/Q3/Q4 文本逐节强化
- [x] §10 灵敏度+一致性扩展
- [x] §11 评价 + §12 结论扩展
- [x] 所有引用与交叉引用解析正常
- [x] 编译通过 0 error / 0 undefined ref
- [x] 关键数值前后一致
- [x] 被禁用语清查通过
- [x] 模型完整 / 四问递进 / QUBO 比特预算清楚 / SDK-CIM 真机证据明确 / Q4 K 灵敏度充分
