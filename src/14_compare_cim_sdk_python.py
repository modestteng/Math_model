"""
真机 / Kaiwu SDK SA / 纯 Python 三方法横向对比分析。
按新规则 §二.7：真机结果归位到 results/真机结果/

输出
  results/真机结果/comparison_summary.json   完整对比数据
  results/真机结果/comparison_table.csv      汇总表
  figures/fig_cim_sdk_python_compare.png     三方法对比柱状图（含可行率 + 加速 + J）
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/真机结果"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False


# ============================================================
# 数据汇总：从已有 JSON 提取所有真机 / SDK / Python 结果
# ============================================================
DATA = []

# ----- Q1 -----
DATA.append(dict(
    problem="Q1",
    method="纯 Python (Held-Karp 精确)",
    description="DP O(2^n n^2) 精确求解",
    n_qubo_vars=None,  # 不用 QUBO
    feasibility_rate=1.0,
    travel=29, penalty=None, J=29,
    time_sec=0.05,
    feasibility="100%（精确）",
    note="全局最优",
))
DATA.append(dict(
    problem="Q1",
    method="Kaiwu SDK SA tuned",
    description="经典模拟退火，5 seeds × 200 chains",
    n_qubo_vars=225,
    feasibility_rate=0.006,  # 6/1000
    travel=29, penalty=None, J=29,  # tuned 后 polish 到 29
    time_sec=3.989,
    feasibility="6/1000 = 0.6%",
    note="tuned + polish 后达到全局最优",
))
DATA.append(dict(
    problem="Q1",
    method="Kaiwu CIM 真机 (R-Q1-004, 16:59)",
    description="CPQC-550, sample_number=10, 8-bit 精度",
    n_qubo_vars=225,
    feasibility_rate=0.10,  # 1/10
    travel=31,  # polish 后
    penalty=None, J=31,
    time_sec=61.99,
    feasibility="1/10 = 10%",
    note="真机原始 cost=69 → polish 31，差全局最优 +2 (6.9%)",
))
DATA.append(dict(
    problem="Q1",
    method="Kaiwu CIM 真机 (R-Q1-005, 19:51)",
    description="CPQC-550, sample_number=10（重测）",
    n_qubo_vars=225,
    feasibility_rate=0.10,
    travel=31, penalty=None, J=31,
    time_sec=63.4,
    feasibility="1/10 = 10%",
    note="重测一致：travel=31，可行率 10% 与首次相同",
))

# ----- Q2 -----
DATA.append(dict(
    problem="Q2",
    method="纯 Python LNS+SA",
    description="多起点 polish + LNS + 多 seed SA",
    n_qubo_vars=None,
    feasibility_rate=1.0,
    travel=31, penalty=84090, J=84121,
    time_sec=85,
    feasibility="100%",
    note="基线（接近全局最优，跨算法稳健）",
))
DATA.append(dict(
    problem="Q2",
    method="Kaiwu SDK SA",
    description="经典 SA，方案 C：one-hot+距离，TW 解码后评估",
    n_qubo_vars=225,
    feasibility_rate=0.006,
    travel=31, penalty=84090, J=84121,
    time_sec=3.84,
    feasibility="6/1000 = 0.6%",
    note="对所有可行解 polish 后取最优，与 Python 完全一致",
))
DATA.append(dict(
    problem="Q2",
    method="Kaiwu CIM 真机 (R-Q2-006, 19:52)",
    description="CPQC-550, sample_number=10",
    n_qubo_vars=225,
    feasibility_rate=0.10,
    travel=31, penalty=84090, J=84121,
    time_sec=122.3,
    feasibility="1/10 = 10%",
    note="与 Python 完全一致（J=84121 跨方法收敛）⭐",
))

# ----- Q3 -----
DATA.append(dict(
    problem="Q3",
    method="纯 Python LNS",
    description="多起点 polish + LNS 500 + SA 3 seeds + 3-opt",
    n_qubo_vars=None,
    feasibility_rate=1.0,
    travel=56, penalty=4941850, J=4941906,
    time_sec=829,
    feasibility="100%",
    note="基线（n=50 单车 + TW）",
))
DATA.append(dict(
    problem="Q3",
    method="Kaiwu SDK SA 分解",
    description="滚动窗口 3 段 × 3 seeds，每子 QUBO ≤289 比特",
    n_qubo_vars="≤289 (per sub)",
    feasibility_rate=0.020,  # 平均
    travel=56, penalty=4941850, J=4941906,
    time_sec=31.44,
    feasibility="0-2.1% (per sub)",
    note="与 Python 完全一致；26× 加速",
))
DATA.append(dict(
    problem="Q3",
    method="Kaiwu CIM 真机（部分丢失）",
    description="CPQC-550；seg1 已提交但脚本 kill 时结果未取回",
    n_qubo_vars="289 (seg1 only)",
    feasibility_rate=None,
    travel=None, penalty=None, J=None,
    time_sec=None,
    feasibility="N/A（结果丢失）",
    note="seg1 配额扣 1 次但结果丢失；seg2/seg3 待用户 web 上传",
))

# ----- Q4 -----
DATA.append(dict(
    problem="Q4",
    method="纯 Python LNS attack",
    description="6 K × 4 seeds × 500 LNS + ref warm start",
    n_qubo_vars=None,
    feasibility_rate=1.0,
    travel=109, penalty=40, J=7149,  # K=7, M=1000 综合目标
    time_sec=1182,
    feasibility="100%",
    note="K=7 主方案，与已知参考解持平",
))
DATA.append(dict(
    problem="Q4",
    method="Kaiwu SDK SA 分解",
    description="每车一子 TSP QUBO，7 个 ≤64 比特",
    n_qubo_vars="≤64 (per vehicle)",
    feasibility_rate=0.027,  # 平均 4/150
    travel=109, penalty=40, J=7149,
    time_sec=3.77,
    feasibility="0-3.3% (per vehicle)",
    note="与 Python 完全一致；313× 加速",
))
DATA.append(dict(
    problem="Q4",
    method="Kaiwu CIM 真机（待补）",
    description="CPQC-550；7 个子 QUBO 待用户 web 上传",
    n_qubo_vars="≤64 (per vehicle)",
    feasibility_rate=None,
    travel=None, penalty=None, J=None,
    time_sec=None,
    feasibility="N/A（待上传）",
    note="待 web 平台一次上传 7 个子矩阵后回填",
))

# ============================================================
# 落盘
# ============================================================
df = pd.DataFrame(DATA)
df_csv = OUT / "comparison_table.csv"
df.to_csv(df_csv, index=False, encoding="utf-8-sig")
print(f"[写出] {df_csv.relative_to(ROOT)}")

(OUT / "comparison_summary.json").write_text(
    json.dumps(DATA, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {(OUT / 'comparison_summary.json').relative_to(ROOT)}")

# 终端汇总打印
print("\n" + "=" * 100)
print("真机 / Kaiwu SDK SA / 纯 Python 三方法横向对比")
print("=" * 100)
for q in ["Q1", "Q2", "Q3", "Q4"]:
    print(f"\n--- {q} ---")
    sub = [d for d in DATA if d["problem"] == q]
    for d in sub:
        J = d["J"]; tr = d["travel"]; pn = d["penalty"]; t = d["time_sec"]
        feas = d["feasibility"]
        line = f"  {d['method']:<35s}  J={J!s:<9} travel={tr!s:<5} pen={pn!s:<10} time={t!s:>7s}s  feas={feas}"
        print(line)

# ============================================================
# 图：三方法对比柱状图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
problems = ["Q1", "Q2", "Q3", "Q4"]
labels = ["Pure Python", "SDK SA", "CIM 真机"]
colors = ["#5cb85c", "#3a78c2", "#d9534f"]

for ax, q in zip(axes.flat, problems):
    sub = [d for d in DATA if d["problem"] == q]
    # 三方法汇总
    method_groups = {
        "Pure Python": next((d for d in sub if "纯 Python" in d["method"]), None),
        "SDK SA": next((d for d in sub if "SDK SA" in d["method"]), None),
        "CIM 真机": next((d for d in sub if "CIM 真机" in d["method"] and d["J"] is not None), None),
    }
    Js = []
    for k in labels:
        d = method_groups.get(k)
        Js.append(d["J"] if d and d["J"] is not None else 0)

    x = np.arange(len(labels))
    bars = ax.bar(x, Js, color=colors, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, Js):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0,
                    "（待真机）", ha="center", va="bottom", fontsize=9, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_yscale("symlog")
    ax.set_ylabel("目标值 J（symlog）")
    title = {"Q1": f"Q1 · n=15 单车（J=travel）",
             "Q2": f"Q2 · n=15 + 时间窗（J=travel+pen）",
             "Q3": f"Q3 · n=50 单车（J=travel+pen）",
             "Q4": f"Q4 · n=50 多车 K=7（J=1000K+travel+pen）"}[q]
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("4 问 × 3 方法横向对比（纯 Python / SDK SA / CIM 真机）",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "fig_cim_sdk_python_compare.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_cim_sdk_python_compare.pdf", bbox_inches="tight")
plt.close(fig)
print(f"[写出] figures/fig_cim_sdk_python_compare.png + .pdf")

# 第二张图：可行率对比
fig, ax = plt.subplots(figsize=(11, 5))
methods_for_feas = []
for q in problems:
    sub = [d for d in DATA if d["problem"] == q and d["feasibility_rate"] is not None]
    for d in sub:
        if "纯 Python" not in d["method"]:  # 启发式 Python 默认 100%，对比无意义
            methods_for_feas.append(dict(
                problem=q, method=d["method"].replace("Kaiwu ", "")[:25],
                feas_rate=d["feasibility_rate"]))
df_feas = pd.DataFrame(methods_for_feas)
xs = np.arange(len(df_feas))
colors_feas = ["#3a78c2" if "SDK" in m else "#d9534f" for m in df_feas["method"]]
bars = ax.bar(xs, df_feas["feas_rate"] * 100, color=colors_feas, edgecolor="black", linewidth=0.5)
for b, v in zip(bars, df_feas["feas_rate"]):
    ax.text(b.get_x() + b.get_width() / 2, v * 100, f"{v*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['problem']}\n{r['method']}" for _, r in df_feas.iterrows()],
                   fontsize=8, rotation=20)
ax.set_ylabel("SA 可行率 (%)")
ax.set_title("SDK SA vs CIM 真机：可行率对比（蓝=SDK，红=CIM）")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_cim_vs_sdk_feasibility.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig_cim_vs_sdk_feasibility.pdf", bbox_inches="tight")
plt.close(fig)
print(f"[写出] figures/fig_cim_vs_sdk_feasibility.png + .pdf")

print("\n[ALL DONE]")
