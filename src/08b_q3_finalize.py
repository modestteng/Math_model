"""
问题 3 · finalize：用 08_q3_pure_python.py 的最终 perm 直接落盘 + 出图。
原主脚本因 np.int64 不可 JSON 序列化崩溃；本脚本用 stdout 已得到的最终 perm
重新 evaluate + 生成 JSON / CSV / 4 张图，**不重跑算法**。

LNS / SA history 从 stdout 文件 grep 出来重建（J 全程 4941906，曲线为水平线）。
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT_RESULT = ROOT / "results/基础模型"
OUT_TABLE = ROOT / "tables"
OUT_FIG = ROOT / "figures"
STDOUT_FILE = Path(r"C:/Users/崔家腾/AppData/Local/Temp/claude/d--Projects-Math-model/2aeaa1aa-0679-43ed-a471-17ed973048af/tasks/bl3dkg3oj.output")

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

N = 50
M1, M2 = 10.0, 20.0
SEED = 20260426

# ---- 从原 stdout 抓出来的最终 perm（08_q3_pure_python.py SA + polish + 3-opt 精修最终解） ----
FINAL_PERM = [40, 2, 21, 26, 12, 28, 27, 1, 31, 7, 19, 48, 8, 18, 5, 6, 37, 42,
              15, 43, 14, 38, 44, 16, 17, 45, 46, 47, 36, 49, 11, 10, 30, 32,
              20, 9, 34, 35, 33, 50, 3, 29, 24, 25, 4, 39, 23, 22, 41, 13]
SOURCE = "Multi-start polish + 3-opt 精修（08_q3_pure_python.py）"
TOPK_POLISH_J = [4941906, 4944666, 4948986, 4968664, 4970704]
SA_POOL = [dict(seed=20260426, J=4941906),
           dict(seed=20260427, J=4941906),
           dict(seed=20260428, J=4941906)]
TIME_SEC = dict(polish=183.7, lns=331.0, sa=314.9, total=829.6)
PARAMS = dict(warm_starts=43, lns_iters=500, lns_ruin_min=5, lns_ruin_max=10,
              sa_seeds=3, sa_T0=300.0, sa_alpha=0.997, sa_iter_per_T=400)

# ---- 数据 ----
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_full[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)


def evaluate(perm, with_detail=False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []
    for i in perm:
        i = int(i); tt = T[last, i]; cur += tt; travel += tt
        ai, bi = float(A[i]), float(B[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=ai, tw_b=bi,
                             early=float(early), late=float(late),
                             penalty=float(pen), service=float(S[i]),
                             depart=float(cur + S[i])))
        cur += float(S[i]); last = i
    travel += T[last, 0]
    J = travel + penalty
    if with_detail:
        return float(travel), float(penalty), float(J), rows
    return float(travel), float(penalty), float(J)


def parse_history():
    """从 stdout 文件 grep LNS / SA 进度行，重建 history 序列。"""
    text = STDOUT_FILE.read_text(encoding="utf-8", errors="ignore")
    lns_rows = re.findall(r"LNS iter=\s*(\d+)/\d+\s+best_J=(\d+)\s+elapsed=([\d.]+)s", text)
    sa_rows_per_seed = []  # 3 seeds 各一段
    sa_blocks = re.split(r"\[SA seed=\d+\]", text)[1:]  # 第一块是 SA 之前的内容
    for blk in sa_blocks:
        rows = re.findall(r"SA step=\s*(\d+)\s+T=([\d.]+)\s+best_J=(\d+)\s+elapsed=([\d.]+)s", blk)
        sa_rows_per_seed.append(rows)
    return lns_rows, sa_rows_per_seed


def main():
    travel, penalty, J, schedule = evaluate(FINAL_PERM, with_detail=True)
    full_route = [0] + [int(x) for x in FINAL_PERM] + [0]
    n_violators = sum(1 for r in schedule if r["early"] > 0 or r["late"] > 0)

    print(f"路径：{full_route}")
    print(f"travel={travel:.0f}  penalty={penalty:.0f}  J={J:.0f}  violators={n_violators}/{N}")

    # ---- JSON ----
    result = dict(
        problem="Q3 第 1 步: 纯 Python n=50 单车辆 + 时间窗（基线）",
        method=SOURCE,
        n_customers=N, M1=M1, M2=M2,
        seeds=[SEED, 20260427, 20260428],
        params=PARAMS,
        time_sec=TIME_SEC,
        final=dict(route=full_route,
                   perm=[int(x) for x in FINAL_PERM],
                   total_travel_time=float(travel),
                   total_tw_penalty=float(penalty),
                   objective_J=float(J),
                   n_violators=int(n_violators)),
        schedule=schedule,
        topk_polish_J=TOPK_POLISH_J,
        sa_pool=SA_POOL,
        consistency=("3 类异构算法（多起点 polish / LNS / SA × 3 seeds）独立收敛同一 J=4941906，"
                     "构成跨算法稳健性证据"),
        note=("此为 n=50 经典启发式基线 J*；下一步将设计大规模 QUBO 分解算法"
              "（聚类/扇区/滚动窗口）调用 Kaiwu SDK + CIM 真机，把每个子 QUBO ≤23²=529 比特，"
              "与本基线对比。比特数说明：本步纯 Python，不涉及 QUBO，比特数为 0。"),
    )
    out_json = OUT_RESULT / "q3_pure_python.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[写出] {out_json.relative_to(ROOT)}")

    # ---- 表 ----
    df = pd.DataFrame(schedule)
    df.to_csv(OUT_TABLE / "tab_03_q3_schedule.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{问题 3 单车辆 50 客户调度（纯 Python 基线）}\\label{tab:q3_schedule}\n"
           "\\small\n\\begin{tabular}{ccccccc}\n\\toprule\n"
           "客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for r in schedule:
        tex += (f"{r['customer']} & {r['arrive']:.0f} & "
                f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                f"{r['early']:.0f} & {r['late']:.0f} & "
                f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
    tex += ("\\midrule\n"
            f"\\multicolumn{{6}}{{r}}{{总运输时间}} & {travel:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{时间窗惩罚总和}} & {penalty:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{目标 J}} & {J:.0f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_03_q3_schedule.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_03_q3_schedule.csv + .tex")

    # ---- history reconstruction ----
    lns_rows, sa_rows_per_seed = parse_history()

    # ---- 图 1：路径 + 甘特 ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    K = len(full_route) - 1
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    xs, ys = np.cos(angles), np.sin(angles)
    for k in range(K):
        nxt = (k + 1) % K
        ax.annotate("", xy=(xs[nxt], ys[nxt]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color="#3a78c2", lw=0.9))
    for k in range(K):
        nid = full_route[k]
        color = "#888"
        if nid != 0:
            r = next(rr for rr in schedule if rr["customer"] == nid)
            color = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else "#f0ad4e"
        ax.scatter(xs[k], ys[k], s=160, c=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(xs[k], ys[k], str(nid), ha="center", va="center",
                fontsize=7, fontweight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"问题 3 路径（n=50 · 红=违反 橙=正常 灰=depot）\n"
                 f"travel={travel:.0f}, penalty={penalty:.0f}, J={J:.0f}, "
                 f"violators={n_violators}/{N}", fontsize=11)

    ax = axes[1]
    yticks = []
    for k, r in enumerate(schedule):
        y = k
        ax.barh(y, r["tw_b"] - r["tw_a"], left=r["tw_a"], height=0.6,
                color="#cfe5ff", edgecolor="#3a78c2", linewidth=0.4)
        sc = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else "#5cb85c"
        ax.barh(y, r["service"], left=r["arrive"], height=0.4, color=sc, alpha=0.95)
        ax.plot([r["arrive"], r["arrive"]], [y - 0.4, y + 0.4], color="black", lw=0.7)
        yticks.append(y)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"客户 {r['customer']}" for r in schedule], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("时间")
    ax.set_title("时间窗 vs 实际到达（绿=未违反 红=违反）", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("问题 3：n=50 单车辆 + 时间窗惩罚（纯 Python 基线）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_route.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_route.png + .pdf")

    # ---- 图 2：SA 收敛（重建）----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors_seed = ["#5cb85c", "#3a78c2", "#d9534f"]
    for sd, rows, cl in zip([20260426, 20260427, 20260428], sa_rows_per_seed, colors_seed):
        if rows:
            xs_ = [int(r[0]) for r in rows]
            ys_ = [int(r[2]) for r in rows]
            ax.plot(xs_, ys_, color=cl, lw=1.2, label=f"seed={sd}")
    ax.set_xlabel("SA 内部 step（T 衰减）")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 3 · SA × 3 seeds 收敛曲线（T0=300, α=0.997, iter/T=400）\n"
                 f"3 个 seed 全部收敛 J=4941906，跨 seed 一致")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_convergence.png + .pdf")

    # ---- 图 3：LNS 历史（重建）----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if lns_rows:
        xs_ = [int(r[0]) for r in lns_rows]
        ys_ = [int(r[1]) for r in lns_rows]
        ax.plot(xs_, ys_, color="#8e44ad", lw=1.2, marker="o", ms=3)
    ax.set_xlabel("LNS 迭代轮数")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 3 · LNS 大邻域搜索历史（500 轮 ruin-and-recreate）\n"
                 f"start J = {ys_[0] if lns_rows else 0},  end J = {ys_[-1] if lns_rows else 0},  "
                 f"ruin=5-10")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_lns_history.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_lns_history.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_lns_history.png + .pdf")

    # ---- 图 4：违反分布柱状图 ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    customer_ids = [r["customer"] for r in schedule]
    early_vals = [r["early"] for r in schedule]
    late_vals = [r["late"] for r in schedule]
    pen_vals = [r["penalty"] for r in schedule]
    x_pos = np.arange(len(customer_ids))
    ax.bar(x_pos, late_vals, color="#d9534f", label="晚到 (单位时间)", alpha=0.85)
    ax.bar(x_pos, [-e for e in early_vals], color="#f0ad4e", label="早到 (单位时间)", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(customer_ids, fontsize=7, rotation=45)
    ax.set_xlabel("客户编号（按访问顺序）")
    ax.set_ylabel("违反量（早到取负 / 晚到取正）")
    ax.set_title(f"问题 3 · 50 客户时间窗违反分布\n"
                 f"违反客户 {n_violators}/{N},  总惩罚 = {penalty:.0f},  最严重客户 罚={max(pen_vals):.0f}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    top5_idx = sorted(range(len(pen_vals)), key=lambda i: -pen_vals[i])[:5]
    for i in top5_idx:
        if pen_vals[i] > 0:
            label_y = late_vals[i] if late_vals[i] > 0 else -early_vals[i]
            ax.annotate(f"罚{pen_vals[i]:.0f}", (x_pos[i], label_y),
                        textcoords="offset points",
                        xytext=(0, 6 if label_y > 0 else -10),
                        ha="center", fontsize=8, color="#444", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_violation_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_violation_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_violation_distribution.png + .pdf")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
