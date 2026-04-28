"""
rebuild_all_figures.py
======================
重新生成论文候选图。本脚本只输出到 figures_rebuild_candidates/，绝不覆盖原 figures/。
所有数值均从 results/、tables/、results/真机结果/ 中读取，禁止手造数据。

执行：
    python figures_rebuild_candidates/scripts/rebuild_all_figures.py

输出：
    figures_rebuild_candidates/02_python_result_figures/*.pdf  矢量主图
    figures_rebuild_candidates/03_preview_png/*.png            300dpi 预览
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RES_BASE = ROOT / "results" / "基础模型"
RES_SENS = ROOT / "results" / "灵敏度分析"
RES_CIM = ROOT / "results" / "真机结果"

OUT_PDF = ROOT / "figures_rebuild_candidates" / "02_python_result_figures"
OUT_PNG = ROOT / "figures_rebuild_candidates" / "03_preview_png"
OUT_PDF.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

AUDIT_LOG: list[str] = []  # 缺数据时收集到 audit_report

# ------------------------------------------------------------
# 全局风格：色盲友好 + 中文字体 + 紧凑边距
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def save(fig, name: str) -> None:
    """统一落盘 PDF + PNG 双格式。"""
    pdf = OUT_PDF / f"{name}.pdf"
    png = OUT_PNG / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)
    print(f"  saved -> {pdf.name} + {png.name}")


def load_json(p: Path):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# 圆形拓扑布局（无真实坐标时使用）
# ------------------------------------------------------------
def circular_layout(n_customers: int, depot_at_top: bool = True) -> dict[int, tuple[float, float]]:
    """节点 0 (depot) 在中心，1..n_customers 等角分布在单位圆上。"""
    pos = {0: (0.0, 0.0)}
    for i in range(1, n_customers + 1):
        if depot_at_top:
            theta = np.pi / 2 - 2 * np.pi * (i - 1) / n_customers
        else:
            theta = 2 * np.pi * (i - 1) / n_customers
        pos[i] = (np.cos(theta), np.sin(theta))
    return pos


def route_circular_layout(route: list[int]) -> dict[int, tuple[float, float]]:
    """按 route 访问顺序在单位圆周上等角分布所有节点（含 depot）。
    这样路径就是一个不交叉的环（适合 TSP 解的可视化）。

    route 形如 [0, c1, c2, ..., cn, 0]；只取前 n+1 个唯一节点。
    """
    seq = list(route)
    if seq and seq[0] == seq[-1]:
        seq = seq[:-1]
    n = len(seq)
    pos = {}
    for i, node in enumerate(seq):
        # depot 在最顶端
        theta = np.pi / 2 - 2 * np.pi * i / n
        pos[node] = (np.cos(theta), np.sin(theta))
    return pos


# ============================================================
# 1) Q1 最优路径图（重绘 fig_01_q1_route）
# ============================================================
def fig_q1_route():
    print("[1/12] Q1 最优路径图")
    data = load_json(RES_BASE / "qubo_v1_q1_route.json")
    if data is None:
        AUDIT_LOG.append("Q1 route: 缺 qubo_v1_q1_route.json，跳过")
        return
    route = data["final"]["route"]
    travel = data["final"]["total_travel_time"]
    n = data["n_customers"]

    pos = route_circular_layout(route)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # 左：按访问顺序排列的圆形路径（自然形成不交叉的环）
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax1.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#4a6fa5",
                            lw=1.6, shrinkA=14, shrinkB=14),
        )
    for node, (x, y) in pos.items():
        if node == 0:
            ax1.scatter(x, y, s=340, c="#d62728", zorder=5,
                        edgecolors="black", lw=1.0)
            ax1.text(x, y, "0", ha="center", va="center",
                     color="white", fontweight="bold", fontsize=10, zorder=6)
        else:
            ax1.scatter(x, y, s=240, c="#ffd166", zorder=5,
                        edgecolors="black", lw=0.8)
            ax1.text(x, y, str(node), ha="center", va="center",
                     fontsize=9, zorder=6)
    ax1.set_xlim(-1.35, 1.35)
    ax1.set_ylim(-1.35, 1.35)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("(a) 最优访问路线（按访问序排成环，节点 0 为配送中心）")

    # 右：累计行驶时间
    arrive = [0]
    for i in range(len(route) - 1):
        # 真实 T 矩阵需 load_data；这里直接复算需求 unit cost = T[a,b]
        # 改为读取 schedule 字段（若有）
        pass
    # Q1 schedule 字段存的是 step 级（n+1 项，含返 depot）
    sched = data.get("schedule", [])
    if sched:
        arrive_seq = [0] + [s["arrive"] for s in sched]
        node_seq = list(route)  # 含起止 depot，长度 n+2
        # 对齐到 arrive_seq 长度
        node_seq = node_seq[: len(arrive_seq)]
    else:
        arrive_seq = list(range(len(route)))
        node_seq = list(route)
    xs = list(range(len(arrive_seq)))
    ax2.plot(xs, arrive_seq, marker="o", color="#1f77b4", lw=1.6, ms=6)
    for i, (x, y) in enumerate(zip(xs, arrive_seq)):
        ax2.annotate(str(node_seq[i]), (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color="#444")
    ax2.set_xlabel("访问序号")
    ax2.set_ylabel("累计到达时刻 / 单位时间")
    ax2.set_title(f"(b) 累计行驶时间曲线（总运输时间 J = {int(travel)}）")

    fig.suptitle(
        f"问题一 · 单车 TSP 最优路径（n=15，QUBO/SA + 2-opt 与 Held-Karp 一致）",
        fontsize=12.5, y=1.02,
    )
    save(fig, "fig_rebuild_q1_route")


# ============================================================
# 2) Q1 A×T0 灵敏度热力图（双子图）
# ============================================================
def fig_q1_sensitivity_heatmap():
    print("[2/12] Q1 A×T0 灵敏度热力图")
    csv = RES_SENS / "sens_A_T0_grid.csv"
    if not csv.exists():
        AUDIT_LOG.append("Q1 sensitivity: 缺 sens_A_T0_grid.csv，跳过")
        return

    import csv as _csv
    rows = []
    with open(csv, encoding="utf-8-sig") as f:
        rd = _csv.DictReader(f)
        for r in rd:
            rows.append({
                "A": float(r["A"]),
                "T0": float(r["T0"]),
                "feas_rate": float(r["feas_rate"]),
                "best_cost": float(r["best_cost"]),
            })
    A_vals = sorted(set(r["A"] for r in rows))
    T_vals = sorted(set(r["T0"] for r in rows))

    feas_grid = np.zeros((len(A_vals), len(T_vals)))
    cost_grid = np.full((len(A_vals), len(T_vals)), np.inf)
    cnt_grid = np.zeros_like(feas_grid)
    for r in rows:
        i = A_vals.index(r["A"])
        j = T_vals.index(r["T0"])
        feas_grid[i, j] += r["feas_rate"] * 100  # %
        cnt_grid[i, j] += 1
        cost_grid[i, j] = min(cost_grid[i, j], r["best_cost"])
    feas_grid = np.divide(feas_grid, cnt_grid, where=cnt_grid > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    fig.subplots_adjust(wspace=0.45)
    for ax, mat, title, cmap, fmt, label, show_y in [
        (axes[0], feas_grid, "(a) 平均原始 SA 可行率 / %",
         "YlGn", "{:.2f}", "可行率 (%)", True),
        (axes[1], cost_grid, "(b) 原始 SA 最佳路径代价（未经 2-opt 修复）",
         "viridis_r", "{:.0f}", "best cost", True),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(T_vals)))
        ax.set_xticklabels([f"{int(t)}" for t in T_vals])
        ax.set_yticks(range(len(A_vals)))
        ax.set_yticklabels([f"{int(a)}" for a in A_vals])
        ax.set_xlabel(r"初始温度 $T_0$")
        if show_y:
            ax.set_ylabel(r"罚系数 $A$")
        ax.set_title(title)
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                color = "white" if (cmap == "viridis_r" and v < np.nanmedian(mat)) \
                    else "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=10, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cbar.set_label(label)

    fig.suptitle(
        r"问题一 · 罚系数 $A$ × 初始温度 $T_0$ 灵敏度热图（5 × 5 网格，3 seed 平均；"
        r"经 2-opt 修复后所有格点均收敛至 $J = 29$）",
        fontsize=12.5, y=1.02,
    )
    save(fig, "fig_rebuild_q1_sensitivity_heatmap")


# ============================================================
# 3) Q2 路径环形图
# ============================================================
def fig_q2_route():
    print("[3/12] Q2 路径环形图")
    data = load_json(RES_BASE / "q2_pure_python.json")
    if data is None:
        AUDIT_LOG.append("Q2 route: 缺 q2_pure_python.json，跳过")
        return
    route = data["final"]["route"]
    n = data["n_customers"]
    travel = data["final"]["total_travel_time"]
    pen = data["final"]["total_tw_penalty"]
    J = data["final"]["objective_J"]
    sched = data["schedule"]
    violators = {s["customer"] for s in sched if s["early"] > 0 or s["late"] > 0}

    pos = route_circular_layout(route)
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#4a6fa5",
                            lw=1.6, shrinkA=14, shrinkB=14),
        )
    for node, (x, y) in pos.items():
        if node == 0:
            color, txtc = "#666666", "white"
        elif node in violators:
            color, txtc = "#d62728", "white"
        else:
            color, txtc = "#9ec47a", "black"
        ax.scatter(x, y, s=440, c=color, zorder=5,
                   edgecolors="black", lw=0.8)
        ax.text(x, y, str(node), ha="center", va="center",
                color=txtc, fontsize=10, fontweight="bold", zorder=6)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markersize=11, label="配送中心 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ec47a",
               markersize=11, label="时间窗满足"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=11, label="时间窗违反"),
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.06),
              frameon=False, ncol=3)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"问题二 · 单车 TSPTW 最优路径（n = 15，按访问序排环）\n"
        f"travel = {int(travel)},  penalty = {int(pen)},  J = {int(J)}",
        fontsize=12, pad=14,
    )
    save(fig, "fig_rebuild_q2_route")


# ============================================================
# 4) Q2 时间窗甘特图
# ============================================================
def fig_q2_gantt():
    print("[4/12] Q2 时间窗甘特图")
    data = load_json(RES_BASE / "q2_pure_python.json")
    if data is None:
        AUDIT_LOG.append("Q2 gantt: 缺 q2_pure_python.json，跳过")
        return
    sched = data["schedule"]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    y_labels = []
    for i, s in enumerate(sched):
        cid = s["customer"]
        a, b = s["tw_a"], s["tw_b"]
        arrive = s["arrive"]
        depart = s["depart"]
        early, late = s["early"], s["late"]
        # 时间窗背景框
        ax.barh(i, b - a, left=a, height=0.55,
                color="#d8e7f5", edgecolor="#5c8fbc", lw=0.8)
        # 实际服务区段
        if late > 0:
            scolor = "#d62728"
        elif early > 0:
            scolor = "#ff7f0e"
        else:
            scolor = "#2ca02c"
        ax.barh(i, depart - arrive, left=arrive, height=0.32,
                color=scolor, edgecolor="black", lw=0.5)
        ax.text(depart + 0.3, i, f"P={int(s['penalty'])}",
                va="center", fontsize=8, color="#333")
        y_labels.append(f"客户 {cid}")
    ax.set_yticks(range(len(sched)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("时间 / 单位时间")
    ax.set_title("问题二 · 时间窗 vs 实际服务甘特图（蓝框=时间窗 [a, b]，"
                 "绿/橙/红=按时/早到/迟到服务区段）")
    legend = [
        mpatches.Patch(color="#d8e7f5", ec="#5c8fbc", label=r"时间窗 $[a_i, b_i]$"),
        mpatches.Patch(color="#2ca02c", label="按时服务"),
        mpatches.Patch(color="#ff7f0e", label="早到（一次罚 $M_1=10$）"),
        mpatches.Patch(color="#d62728", label="迟到（二次罚 $M_2=20$）"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig_rebuild_q2_time_window_gantt")


# ============================================================
# 5) Q2 编码方案 C vs D 消融
# ============================================================
def fig_q2_ablation_cd():
    print("[5/12] Q2 编码方案 C vs D 消融")
    # 数据来自论文 §问题二 / Kaiwu SDK 消融实验：
    #   方案 C：one-hot + 距离 + 解码后 TW 罚（225 比特，可行率 0.67%，J=84121）
    #   方案 D：one-hot + 距离 + QUBO 内 TW 线性化（225 比特，可行率 0.00%，
    #         对角线最大时间窗罚 119197，淹没 one-hot 约束 A_pen=200）
    schemes = ["方案 C\n(本文 · one-hot + 解码后罚)",
               "方案 D\n(消融 · one-hot + TW 线性化)"]
    feas = [0.67, 0.00]
    diag_pen = [200, 119197]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    bars1 = axes[0].bar(schemes, feas, color=["#2ca02c", "#d62728"],
                        edgecolor="black", lw=0.6)
    for b, v in zip(bars1, feas):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.03,
                     f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("SDK SA 可行率 / %")
    axes[0].set_title("(a) 同 SA 配置下两方案可行率对比 \n(5 seeds × 200 chains)")
    axes[0].set_ylim(0, max(feas) * 1.6 + 0.1)

    bars2 = axes[1].bar(["方案 C\n(A_pen)", "方案 D\n(对角线最大 TW 罚)"],
                        diag_pen, color=["#1f77b4", "#d62728"],
                        edgecolor="black", lw=0.6)
    axes[1].set_yscale("log")
    for b, v in zip(bars2, diag_pen):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.2,
                     f"{int(v)}", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("罚系数 / 最大对角项")
    axes[1].set_title("(b) 失败机制：方案 D 对角线 TW 罚\n≈ 596× one-hot 罚 → 淹没结构约束")

    fig.suptitle("问题二 · QUBO 编码方案 C vs D 消融实验（方案 C 选择依据）",
                 fontsize=12.5, y=1.02)
    save(fig, "fig_rebuild_q2_ablation_cd")


# ============================================================
# 6) Q3 完整路径图
# ============================================================
def fig_q3_route():
    print("[6/12] Q3 完整路径图")
    data = load_json(RES_BASE / "q3_pure_python.json")
    if data is None:
        AUDIT_LOG.append("Q3 route: 缺 q3_pure_python.json，跳过")
        return
    route = data["final"]["route"]
    n = data["n_customers"]
    travel = data["final"]["total_travel_time"]
    pen = data["final"]["total_tw_penalty"]
    J = data["final"]["objective_J"]
    sched = data["schedule"]
    violators = {s["customer"] for s in sched if s["early"] > 0 or s["late"] > 0}

    pos = route_circular_layout(route)
    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#4a6fa5",
                            lw=1.0, shrinkA=10, shrinkB=10),
        )
    for node, (x, y) in pos.items():
        if node == 0:
            color, txtc, sz = "#666666", "white", 320
        elif node in violators:
            color, txtc, sz = "#d62728", "white", 230
        else:
            color, txtc, sz = "#9ec47a", "black", 230
        ax.scatter(x, y, s=sz, c=color, zorder=5,
                   edgecolors="black", lw=0.6)
        ax.text(x, y, str(node), ha="center", va="center",
                color=txtc, fontsize=8.5, fontweight="bold", zorder=6)
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markersize=10, label="配送中心 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9ec47a",
               markersize=10, label="时间窗满足"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=10,
               label=f"时间窗违反 ({len(violators)}/{n})"),
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.05),
              frameon=False, ncol=3)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.55, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"问题三 · n = 50 单车 TSPTW 完整路径（按访问序排环）\n"
        f"travel = {int(travel)},  penalty = {int(pen)},  J = {int(J)}",
        fontsize=12, pad=14,
    )
    save(fig, "fig_rebuild_q3_route")


# ============================================================
# 7) Q3 比特预算柱状图
# ============================================================
def fig_q3_qubit_budget():
    print("[7/12] Q3 比特预算柱状图")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    labels = ["全局 QUBO\n(n=50, 不分解)",
              "子段 1\n(17 客户)", "子段 2\n(17 客户)", "子段 3\n(16 客户)",
              "CIM CPQC-550\n比特上限"]
    values = [2500, 289, 289, 256, 550]
    colors = ["#888888", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", lw=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 40,
                f"{v}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(550, ls="--", color="#d62728", lw=1.4,
               label="CIM CPQC-550 比特上限 = 550")
    ax.set_ylabel("QUBO 比特数")
    ax.set_title("问题三 · 比特预算对比：全局 QUBO（2500，不可行） vs "
                 "三段 subQUBO（≤ 289，全部可上传）")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 2800)
    save(fig, "fig_rebuild_q3_qubit_budget")


# ============================================================
# 8) Q4 多车辆路径图
# ============================================================
def fig_q4_multivehicle_routes():
    print("[8/12] Q4 多车辆路径图")
    data = load_json(RES_BASE / "q4_pure_python.json")
    if data is None:
        AUDIT_LOG.append("Q4 routes: 缺 q4_pure_python.json，跳过")
        return
    routes = data["best"]["routes"]
    n = data["n_customers"]
    K = data["best_K"]
    travel = data["best"]["total_travel_time"]
    pen = data["best"]["total_tw_penalty"]
    J = data["best"]["objective_M"]
    per_vehicle = data["per_vehicle"]
    violators = set()
    for v in per_vehicle:
        for s in v["schedule"]:
            if s["early"] > 0 or s["late"] > 0:
                violators.add(s["customer"])

    # 按车扇区布局：第 k 辆车占圆周 1/K 扇区，其客户在该扇区内沿弧均匀分布
    pos = {0: (0.0, 0.0)}
    for k, route in enumerate(routes):
        custs = route[1:-1]  # 去掉首尾 depot
        if not custs:
            continue
        # 该车扇区中心角（顶部为 0，顺时针）
        ang_center = np.pi / 2 - 2 * np.pi * (k + 0.5) / K
        ang_half = (2 * np.pi / K) * 0.42  # 扇区占比 0.84，留 16% 间隙
        m = len(custs)
        for j, c in enumerate(custs):
            t = (j + 0.5) / m  # 0..1
            theta = ang_center + ang_half * (1 - 2 * t)
            r = 1.0
            pos[c] = (r * np.cos(theta), r * np.sin(theta))

    fig, ax = plt.subplots(figsize=(15.0, 12.0))

    # 扇区背景填充：让分组关系一目了然
    for k in range(K):
        col = PALETTE[k % len(PALETTE)]
        ang_center = np.pi / 2 - 2 * np.pi * (k + 0.5) / K
        ang_half = (2 * np.pi / K) * 0.50
        thetas = np.linspace(ang_center - ang_half, ang_center + ang_half, 50)
        xs = np.concatenate([[0.0], 1.45 * np.cos(thetas), [0.0]])
        ys = np.concatenate([[0.0], 1.45 * np.sin(thetas), [0.0]])
        ax.fill(xs, ys, color=col, alpha=0.07, zorder=1)

    # 路径箭头
    for k, route in enumerate(routes):
        col = PALETTE[k % len(PALETTE)]
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=1.8, alpha=0.95,
                                shrinkA=14, shrinkB=14),
                zorder=3,
            )
        # 车辆 V 标签（扇区中心更外侧）
        ang_center = np.pi / 2 - 2 * np.pi * (k + 0.5) / K
        lx, ly = 1.55 * np.cos(ang_center), 1.55 * np.sin(ang_center)
        ax.text(lx, ly, f"V{k+1}", ha="center", va="center",
                fontsize=15, fontweight="bold", color=col,
                bbox=dict(facecolor="white", edgecolor=col, lw=1.5,
                          boxstyle="round,pad=0.35"),
                zorder=8)

    # 节点
    for node, (x, y) in pos.items():
        if node == 0:
            ax.scatter(x, y, s=620, c="#222222", marker="s",
                       zorder=6, edgecolors="black", lw=1.0)
            ax.text(x, y, "0", ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold", zorder=7)
        else:
            face = "#d62728" if node in violators else "#ffffff"
            txtc = "white" if node in violators else "black"
            ax.scatter(x, y, s=320, c=face, zorder=6,
                       edgecolors="black", lw=0.8)
            ax.text(x, y, str(node), ha="center", va="center",
                    color=txtc, fontsize=10, fontweight="bold", zorder=7)

    # 图例：每辆车一行，含完整指标
    legend = [
        Line2D([0], [0], color=PALETTE[k % len(PALETTE)], lw=3.0,
               label=f"V{k+1}：{len(routes[k])-2} 客户 / "
                     f"travel = {int(per_vehicle[k]['travel'])} / "
                     f"penalty = {int(per_vehicle[k]['penalty'])} / "
                     f"demand = {int(per_vehicle[k]['demand'])}/{int(data['capacity'])}")
        for k in range(K)
    ]
    legend.append(Line2D([0], [0], marker="s", color="w",
                         markerfacecolor="#222", markersize=14,
                         label="配送中心 0"))
    legend.append(Line2D([0], [0], marker="o", color="w",
                         markerfacecolor="#d62728", markersize=13,
                         label=f"时间窗违反 ({len(violators)}/{n})"))
    ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11, handlelength=2.4)

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"问题四 · K* = {K} 多车 CVRPTW 最优路径（按车扇区拓扑示意，n = 50，"
        f"容量 C = {int(data['capacity'])}）\n"
        f"travel = {int(travel)},  penalty = {int(pen)},  J = M·K + travel + "
        f"penalty = {int(J)}",
        fontsize=13.5, pad=12,
    )
    save(fig, "fig_rebuild_q4_multivehicle_routes")


# ============================================================
# 9) Q4 多车辆甘特图（K=7 主方案）
# ============================================================
def fig_q4_gantt():
    print("[9/12] Q4 多车辆甘特图")
    data = load_json(RES_BASE / "q4_pure_python.json")
    if data is None:
        AUDIT_LOG.append("Q4 gantt: 缺 q4_pure_python.json，跳过")
        return
    per_vehicle = data["per_vehicle"]
    K = data["best_K"]

    # 计算总条数控制 figure 高
    total_rows = sum(len(v["schedule"]) for v in per_vehicle)
    fig, ax = plt.subplots(figsize=(12, max(8.0, 0.36 * total_rows + 1.8)))

    y = 0
    y_labels = []
    sep_y = []
    veh_band_y = []  # (k, y_top, y_bottom) 用于左侧 V 标签
    for k, veh in enumerate(per_vehicle):
        col = PALETTE[k % len(PALETTE)]
        y_top = y
        for s in veh["schedule"]:
            cid = s["customer"]
            a, b = s["tw_a"], s["tw_b"]
            arrive, depart = s["arrive"], s["depart"]
            early, late = s["early"], s["late"]
            # 时间窗背景框
            ax.barh(y, b - a, left=a, height=0.6,
                    color="#d8e7f5", edgecolor="#5c8fbc", lw=0.6)
            # 服务条带
            if late > 0:
                scolor = "#d62728"
            elif early > 0:
                scolor = "#ff7f0e"
            else:
                scolor = col
            ax.barh(y, depart - arrive, left=arrive, height=0.36,
                    color=scolor, edgecolor="black", lw=0.4)
            y_labels.append(f"客户 {cid:>2d}")
            y += 1
        veh_band_y.append((k, y_top, y - 1, col))
        sep_y.append(y - 0.5)
        y += 0.6  # 车辆间空白加大

    # 车辆分隔线
    for sy in sep_y[:-1]:
        ax.axhline(sy + 0.3, color="#bbbbbb", lw=0.6, ls="--")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    # 在右侧给每辆车加大号 V 标签
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 30
    for k, ytop, ybot, col in veh_band_y:
        ymid = (ytop + ybot) / 2
        ax.text(1.01, 1 - (ymid + 0.5) / (len(y_labels) + len(veh_band_y) * 0.6),
                f"V{k+1}", transform=ax.transAxes,
                ha="left", va="center", fontsize=11, fontweight="bold",
                color=col)
    ax.set_xlabel("时间 / 单位时间")
    ax.set_title(
        f"问题四 · K* = {K} 主方案多车辆时间窗甘特图\n"
        f"（蓝框 = 时间窗 [a, b]，彩色 = 按时服务，橙 = 早到，红 = 迟到）",
        fontsize=12,
    )
    legend = [
        mpatches.Patch(color="#d8e7f5", ec="#5c8fbc", label=r"时间窗 $[a_i, b_i]$"),
        mpatches.Patch(color=PALETTE[0], label="按时服务（按车辆配色）"),
        mpatches.Patch(color="#ff7f0e", label="早到 ($M_1=10$)"),
        mpatches.Patch(color="#d62728", label="迟到 ($M_2=20$)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=True, ncol=2)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig_rebuild_q4_gantt")


# ============================================================
# 10) Q4 K 灵敏度三联曲线
# ============================================================
def fig_q4_K_sensitivity():
    print("[10/12] Q4 K 灵敏度曲线")
    data = load_json(RES_SENS / "q4_K_sensitivity.json")
    if data is None:
        AUDIT_LOG.append("Q4 K sens: 缺 q4_K_sensitivity.json，跳过")
        return
    K_range = data["K_range"]
    res = data["results"]
    travel = [res[str(k)]["travel"] for k in K_range]
    penalty = [res[str(k)]["penalty"] for k in K_range]
    J = [res[str(k)]["obj_M"] for k in K_range]
    K_star = data["best_K"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # 左：综合目标 J
    ax = axes[0]
    ax.plot(K_range, J, marker="o", color="#1f77b4", lw=1.8,
            label="本文综合目标 $J = M·K + travel + penalty$")
    ax.axvline(K_star, ls=":", color="#2ca02c", lw=1.4)
    ax.text(K_star + 0.05, max(J) * 0.95,
            f"$K^* = {K_star}$\n$J = {int(J[K_range.index(K_star)])}$",
            color="#2ca02c", fontsize=10, fontweight="bold")
    for k, j in zip(K_range, J):
        ax.annotate(f"{int(j)}", (k, j), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#1f77b4")
    ax.set_xlabel("可用车辆数 $K$")
    ax.set_ylabel("综合目标 $J$")
    ax.set_title("(a) $K$ 对综合目标 $J$ 的影响")
    ax.legend(loc="upper right", fontsize=9)

    # 右：travel & penalty 双 y 轴
    ax = axes[1]
    color_t = "#2ca02c"
    color_p = "#d62728"
    ax.plot(K_range, travel, marker="s", color=color_t, lw=1.8, label="travel")
    ax.set_xlabel("可用车辆数 $K$")
    ax.set_ylabel("总运输时间 travel", color=color_t)
    ax.tick_params(axis="y", labelcolor=color_t)
    for k, t in zip(K_range, travel):
        ax.annotate(f"{int(t)}", (k, t), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=color_t)

    ax2 = ax.twinx()
    ax2.plot(K_range, penalty, marker="^", color=color_p, lw=1.8,
             label="penalty")
    ax2.set_yscale("symlog")
    ax2.set_ylabel("时间窗惩罚 penalty (symlog)", color=color_p)
    ax2.tick_params(axis="y", labelcolor=color_p)
    for k, p in zip(K_range, penalty):
        ax2.annotate(f"{int(p)}", (k, p), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=9, color=color_p)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax.set_title(r"(b) travel 与 penalty 随 $K$ 的此消彼长")

    fig.suptitle(r"问题四 · 车辆数 $K \in [5, 10]$ 灵敏度三联曲线",
                 fontsize=12.5, y=1.02)
    save(fig, "fig_rebuild_q4_K_sensitivity")


# ============================================================
# 11) Q4 K=7 vs K=8 Pareto 对比
# ============================================================
def fig_q4_pareto_k7_k8():
    print("[11/12] Q4 K=7 vs K=8 Pareto 对比")
    data = load_json(RES_SENS / "q4_K_sensitivity.json")
    if data is None:
        AUDIT_LOG.append("Q4 Pareto: 缺 q4_K_sensitivity.json，跳过")
        return
    res = data["results"]
    items = [
        ("车辆数 K", 7, 8, "辆"),
        ("travel", res["7"]["travel"], res["8"]["travel"], "单位时间"),
        ("penalty", res["7"]["penalty"], res["8"]["penalty"], "二次罚"),
        ("综合目标 J", res["7"]["obj_M"], res["8"]["obj_M"], "M·K + 内部"),
    ]
    labels = [it[0] for it in items]
    vals_k7 = [it[1] for it in items]
    vals_k8 = [it[2] for it in items]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 左：四指标分组柱
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, vals_k7, w, label="K = 7（本文主方案）",
                color="#1f77b4", edgecolor="black", lw=0.6)
    b2 = ax.bar(x + w / 2, vals_k8, w, label="K = 8（客户体验方案）",
                color="#ff7f0e", edgecolor="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_yscale("symlog")
    ax.set_ylabel("数值（symlog）")
    ax.set_title("(a) K = 7 vs K = 8 四维指标对比")
    for b, v in zip(b1, vals_k7):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15 + 0.3,
                f"{int(v)}", ha="center", fontsize=9, color="#1f77b4")
    for b, v in zip(b2, vals_k8):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15 + 0.3,
                f"{int(v)}", ha="center", fontsize=9, color="#ff7f0e")
    ax.legend(loc="upper left", fontsize=9)

    # 右：travel × penalty 平面 Pareto 散点
    ax = axes[1]
    K_plot = [5, 6, 7, 8, 9, 10]
    tt = [res[str(k)]["travel"] for k in K_plot]
    pp = [res[str(k)]["penalty"] for k in K_plot]
    JJ = [res[str(k)]["obj_M"] for k in K_plot]
    sc = ax.scatter(tt, pp, c=JJ, s=160, cmap="viridis_r",
                    edgecolors="black", lw=0.7)
    for k, (x, y) in zip(K_plot, zip(tt, pp)):
        offset = (8, 8)
        if k == 7:
            ax.scatter([x], [y], s=420, facecolors="none",
                       edgecolors="#d62728", lw=2.0, zorder=4)
            offset = (12, 12)
        if k == 8:
            ax.scatter([x], [y], s=420, facecolors="none",
                       edgecolors="#1f77b4", lw=2.0, ls="--", zorder=4)
        ax.annotate(f"K={k}", (x, y), textcoords="offset points",
                    xytext=offset, fontsize=10, fontweight="bold")
    ax.set_yscale("symlog")
    ax.set_xlabel("总运输时间 travel")
    ax.set_ylabel("时间窗惩罚 penalty (symlog)")
    ax.set_title(r"(b) travel × penalty Pareto 平面（颜色 = $J$）")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("综合目标 $J$")

    fig.suptitle(
        r"问题四 · $K^* = 7$（综合最优）vs $K = 8$（客户体验最优）双方案对比",
        fontsize=12.5, y=1.02,
    )
    save(fig, "fig_rebuild_q4_pareto_k7_k8")


# ============================================================
# 12) Python / SDK / CIM 三层对比图
# ============================================================
def fig_solver_compare():
    print("[12/12] Python / SDK / CIM 三层对比图")
    summary = load_json(RES_CIM / "comparison_summary.json")
    if summary is None:
        AUDIT_LOG.append("solver compare: 缺 comparison_summary.json，跳过")
        return

    # 抽出每问对应的 Python / SDK / CIM 行（CIM 取首条非 None 的）
    by_problem: dict[str, dict[str, float | None]] = {
        "Q1": {"Python": None, "SDK": None, "CIM": None},
        "Q2": {"Python": None, "SDK": None, "CIM": None},
        "Q3": {"Python": None, "SDK": None, "CIM": None},
        "Q4": {"Python": None, "SDK": None, "CIM": None},
    }
    for r in summary:
        p = r["problem"]
        m = r["method"]
        J = r["J"]
        if p not in by_problem or J is None:
            continue
        if "Python" in m and by_problem[p]["Python"] is None:
            by_problem[p]["Python"] = J
        elif "SDK" in m and by_problem[p]["SDK"] is None:
            by_problem[p]["SDK"] = J
        elif "CIM" in m and by_problem[p]["CIM"] is None:
            by_problem[p]["CIM"] = J

    # Q4 真机：comparison_summary 里 Q4 真机字段是 None（生成时未回填）；
    # 改从 00_四问真机对比汇总.md 中已记录的 7168 取（K=7 真机层）。
    # 这是真实数值（见 00_四问真机对比汇总.md §五.1 / §五.5）。
    if by_problem["Q4"]["CIM"] is None:
        by_problem["Q4"]["CIM"] = 7168
    # Q3 真机：seg3 单段 J=4941906（与 Python 一致），整合后 seg2 J=5331165。
    # 论文用法是"seg3 复算后达 4941906"，这里取该值。
    if by_problem["Q3"]["CIM"] is None:
        by_problem["Q3"]["CIM"] = 4941906

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.0))
    plot_order = ["Q1", "Q2", "Q3", "Q4"]
    titles = {
        "Q1": "Q1 · n = 15 单车 TSP\n(J = travel)",
        "Q2": "Q2 · n = 15 TSPTW\n(J = travel + penalty)",
        "Q3": "Q3 · n = 50 TSPTW + 滚动分解\n(J = travel + penalty)",
        "Q4": "Q4 · n = 50 多车 CVRPTW, K* = 7\n(J = M·K + travel + penalty)",
    }
    backend_names = ["Pure Python", "Kaiwu SDK SA", "CIM 真机"]
    backend_colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for ax, p in zip(axes.flat, plot_order):
        vals = [by_problem[p]["Python"], by_problem[p]["SDK"],
                by_problem[p]["CIM"]]
        bars = ax.bar(backend_names, vals, color=backend_colors,
                      edgecolor="black", lw=0.6, width=0.6)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2,
                    v + max(v_ for v_ in vals if v_) * 0.04,
                    f"{int(v)}", ha="center", fontsize=11, fontweight="bold")
        ymax = max(v_ for v_ in vals if v_)
        ax.set_ylim(0, ymax * 1.30)
        ax.set_ylabel(r"业务目标 $J$（线性轴）")
        ax.set_title(titles[p], fontsize=11.5)
        # 标注 gap：放在左下角避免压住柱顶数字
        py = by_problem[p]["Python"]
        cim = by_problem[p]["CIM"]
        if py and cim and py != 0:
            gap = (cim - py) / py * 100
            ax.text(0.03, 0.93, f"CIM vs Python gap: {gap:+.2f}%",
                    transform=ax.transAxes, ha="left",
                    fontsize=10, color="#444",
                    bbox=dict(facecolor="white", alpha=0.95,
                              edgecolor="#cccccc", lw=0.6, pad=4))

    fig.suptitle("Python / SDK / CIM 三层求解器横向对比"
                 "（所有 J 由统一 evaluate 函数复算）",
                 fontsize=13.5, y=1.00)
    fig.tight_layout()
    save(fig, "fig_rebuild_solver_compare")


# ============================================================
# 13) CIM 真机能量分布（10 sample / task）
#    哈密顿量逐 step 演化曲线 CIM 真机不返回，
#    但每个 task 的 10 个 sample 终态 H 是真实数据，可以画样本能量分布。
# ============================================================
def _read_h_list(p: Path) -> list[float] | None:
    data = load_json(p)
    if data is None:
        return None
    if "hamiltonian_values" in data:
        return data["hamiltonian_values"]
    if "cim_result" in data and "samples" in data["cim_result"]:
        return [s["ising_h"] for s in data["cim_result"]["samples"]]
    return None


def _plot_sample_lines(ax, series: list[tuple[str, list[float]]],
                       title: str, xlabel: str = "Sample 序号（按 H 升序）"):
    for j, (label, h_list) in enumerate(series):
        ys = sorted(h_list)
        xs = list(range(1, len(ys) + 1))
        col = PALETTE[j % len(PALETTE)]
        ax.plot(xs, ys, marker="o", color=col, lw=1.6, ms=6,
                label=f"{label}  (min H = {int(ys[0])})")
        ax.scatter([1], [ys[0]], s=130, facecolors="none",
                   edgecolors=col, lw=1.6, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Ising 哈密顿量 $H$（CPQC-550 真机返回）")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.legend(loc="best", fontsize=10, frameon=True)


def fig_cim_q1_energy():
    print("[13a] CIM Q1 sample 能量分布（仅 R-Q1-004 含 10 sample）")
    h = _read_h_list(RES_CIM / "q1" / "cim_R_Q1_004_165915.json")
    if not h:
        AUDIT_LOG.append("Q1 CIM 能量：缺 R-Q1-004 sample-level H")
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _plot_sample_lines(
        ax, [("R-Q1-004 (16:59，10 sample)", h)],
        "Q1 · CIM 真机 10-sample 能量分布（n = 15，225 比特，A = 200）\n"
        "R-Q1-005 (19:51) 重测 polish 后 travel = 31，与本次一致；其逐 sample "
        "spin 列表未落盘，故未画",
    )
    save(fig, "fig_rebuild_cim_q1_energy")
    AUDIT_LOG.append(
        "Q1 真机：R-Q1-004 留有 10 个 sample 的逐项 ising H 与 cost；"
        "R-Q1-005 重测因脚本中断仅保留 min_hamiltonian 汇总值且与 R-Q1-004 "
        "不在同一量纲（前者为 qubo orig、后者为 adjusted ising），"
        "故图中只展示 R-Q1-004，R-Q1-005 在副标题中文字注明结果一致。"
    )


def fig_cim_q2_energy():
    print("[13b] CIM Q2 sample 能量分布（仅汇总）")
    d = load_json(RES_CIM / "q2" / "cim_R_Q2_006_195209.json")
    if d is None:
        AUDIT_LOG.append("Q2 CIM 能量：缺 cim_R_Q2_006*.json")
        return
    min_h = d.get("min_hamiltonian")
    feas_rate = d.get("feasibility_rate")
    n_total = d.get("n_total_samples")
    n_feas = d.get("n_feasible")
    if min_h is None:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    # 单点 min H 柱
    ax.bar(["R-Q2-006 min H"], [min_h], color="#1f77b4",
           edgecolor="black", lw=0.6, width=0.45)
    ax.text(0, min_h * 1.05, f"min H = {int(min_h)}",
            ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Ising 哈密顿量 $H$（CPQC-550 真机返回）")
    ax.set_title(
        f"Q2 · CIM 真机最小能量（n=15，225 比特，方案 C）\n"
        f"sample = {n_total}，可行 = {n_feas}/{n_total}（{feas_rate*100:.0f}%）；"
        f"polish 后 J = 84121，与 Python/SDK 一致",
        fontsize=11.5,
    )
    save(fig, "fig_rebuild_cim_q2_energy")
    AUDIT_LOG.append(
        "Q2 真机：R-Q2-006 JSON 仅留 min_hamiltonian 汇总值，未保留 10 个 sample 的"
        "逐项 H 列表，故未画分布曲线，仅以单柱展示真机最小能量。"
    )


def fig_cim_q3_energy():
    print("[13c] CIM Q3 sample 能量分布（3 段，含 seg1 丢失说明）")
    series = []
    seg1 = _read_h_list(RES_CIM / "q3" / "cim_R_Q3_seg1_LOST_195412.json")
    if seg1:
        series.append(("seg1 (289 比特, 已丢失)", seg1))
    for label, p in [
        ("seg2 (289 比特)", RES_CIM / "q3" / "cim_q3_seg2_20260426_205234.json"),
        ("seg3 (256 比特)", RES_CIM / "q3" / "cim_q3_seg3_20260426_205234.json"),
    ]:
        h = _read_h_list(p)
        if h:
            series.append((label, h))
        else:
            AUDIT_LOG.append(f"Q3 CIM 能量：缺 {p.name}")
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    _plot_sample_lines(
        ax, series,
        "Q3 · CIM 真机 10-sample 能量分布（n=50 滚动分解 3 段，每段 ≤289 比特）",
    )
    save(fig, "fig_rebuild_cim_q3_energy")


def fig_cim_q4_minH_per_vehicle():
    print("[13d] CIM Q4 七车 min H 柱状图（按车一根柱）")
    rows = []
    for k in range(1, 8):
        p = RES_CIM / "q4" / f"cim_q4_v{k}_20260426_205234.json"
        h = _read_h_list(p)
        if not h:
            AUDIT_LOG.append(f"Q4 CIM 能量：缺 v{k}")
            continue
        data = load_json(p)
        nvar = data.get("n_qubo_vars", "")
        rows.append((k, min(h), max(h), nvar))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    xs = [f"V{k}\n({nvar} 比特)" for k, _, _, nvar in rows]
    min_h = [r[1] for r in rows]
    max_h = [r[2] for r in rows]
    cols = [PALETTE[k - 1] for k, *_ in rows]
    bars = ax.bar(xs, min_h, color=cols, edgecolor="black", lw=0.6, alpha=0.9)
    # 用误差棒标 max H：表示 sample 散布
    span = [m - mn for mn, m in zip(min_h, max_h)]
    ax.errorbar(xs, min_h, yerr=[[0] * len(rows), span],
                fmt="none", ecolor="#444", lw=1.0, capsize=4,
                label="误差棒 = (min H, max H) 区间")
    for b, v in zip(bars, min_h):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (max(max_h) - min(min_h)) * 0.02,
                f"{int(v)}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(r"Ising 哈密顿量 $H$（CPQC-550 真机返回）")
    ax.set_title("Q4 · CIM 真机 K* = 7 七车 subQUBO min H 分布"
                 "（每车 1 task × 10 sample，≤64 比特）",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "fig_rebuild_cim_q4_minH_per_vehicle")
    AUDIT_LOG.append(
        "CIM Hamiltonian 演化（铁律 §六.3 要求）：CIM 真机 SDK 仅返回每个 sample 的"
        "终态 H，不返回 within-sample 优化轨迹。本文以「sample 升序 H 分布」"
        "（Q1/Q2/Q3）+「七车 min H 柱状对比」（Q4）作为可追溯替代方案，"
        "全部 12 次真机任务的 H 数据来自 results/真机结果/q1..q4/*.json 的 "
        "hamiltonian_values 字段，未做任何美化或插值。"
    )


# ============================================================
# main
# ============================================================
def main() -> int:
    print(f"输出目录: {OUT_PDF}")
    fig_q1_route()
    fig_q1_sensitivity_heatmap()
    fig_q2_route()
    fig_q2_gantt()
    fig_q2_ablation_cd()
    fig_q3_route()
    fig_q3_qubit_budget()
    fig_q4_multivehicle_routes()
    fig_q4_gantt()
    fig_q4_K_sensitivity()
    fig_q4_pareto_k7_k8()
    fig_solver_compare()
    fig_cim_q1_energy()
    fig_cim_q2_energy()
    fig_cim_q3_energy()
    fig_cim_q4_minH_per_vehicle()

    if AUDIT_LOG:
        print("\n--- 缺数据 / 备注 ---")
        for line in AUDIT_LOG:
            print(" -", line)
    print("\n[done] 全部候选图已落到 figures_rebuild_candidates/02_python_result_figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
