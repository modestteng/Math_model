"""
问题 2：考虑时间窗惩罚、暂不考虑容量约束的单车辆调度（纯 Python 求解）

模型
----
  设客户集合 N={1..n}（本问 n=15），depot=0；π=(π1,..,πn) 为访问排列。
  无空闲等待：t_{π_1} = T[0,π_1]，t_{π_k} = t_{π_{k-1}} + s_{π_{k-1}} + T[π_{k-1},π_k]。
  时间窗惩罚 P_i = M1·([a_i-t_i]^+)^2 + M2·([t_i-b_i]^+)^2，M1=10, M2=20。
  目标 J(π) = 总运输时间(π) + Σ_i P_i(π) → min。

求解策略（纯 Python，独立于 Kaiwu SDK）
----
  (1) 6 种 warm start：随机 + 2 种 NN（按 T 邻近 / 按时间窗下界）+ 按 a_i / 按 b_i 升序 + 按 (a_i+b_i)/2；
  (2) 多起点局部搜索：2-opt + Or-opt(1/2/3) + swap，按 J 反复轮转直到双稳；
  (3) 模拟退火 (SA on permutation)：邻域 swap / insert-segment / reverse，温度衰减；
  (4) 取最优解，输出每客户到达时刻、违反程度、惩罚值与总目标。

输出
----
  results/基础模型/q2_pure_python.json
  tables/tab_02_q2_schedule.csv (+ .tex)
  figures/fig_02_q2_route.png  (+ .pdf)
  figures/fig_02_q2_convergence.png (+ .pdf)
"""
from __future__ import annotations
import json
import time
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
for p in (OUT_RESULT, OUT_TABLE, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

N = 15
M1, M2 = 10.0, 20.0
SEED = 20260426

# ------------------------------------------------------------
# 1. 数据
# ------------------------------------------------------------
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
assert T_full.shape == (51, 51)

T = T_full[: N + 1, : N + 1].astype(float)
nodes = nodes_raw.iloc[: N + 1].copy()
A = nodes["tw_a"].values.astype(float)   # depot a=0, b=30 (we still ignore depot tw on objective)
B = nodes["tw_b"].values.astype(float)
S = nodes["service"].values.astype(float)
# 注意：客户索引从 1 开始；A[0],B[0],S[0] 为 depot 自身（不计入惩罚）

# ------------------------------------------------------------
# 2. 评估函数（含逐客户调度）
# ------------------------------------------------------------
def evaluate(perm, with_detail: bool = False):
    """perm: 长度 n 的客户访问序列（1..n 的一种排列）。
    返回 (travel, penalty, J) 或 (..., schedule rows)。"""
    travel = 0.0
    penalty = 0.0
    cur = 0.0
    last = 0
    rows = []
    for i in perm:
        tt = T[last, i]
        cur += tt
        travel += tt
        ai, bi = A[i], B[i]
        early = max(0.0, ai - cur)
        late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen
        if with_detail:
            rows.append(dict(
                customer=int(i),
                arrive=float(cur),
                tw_a=float(ai), tw_b=float(bi),
                early=float(early), late=float(late),
                penalty=float(pen),
                service=float(S[i]),
                depart=float(cur + S[i]),
            ))
        cur += S[i]
        last = i
    travel += T[last, 0]  # 回 depot
    J = travel + penalty
    if with_detail:
        return travel, penalty, J, rows
    return travel, penalty, J


# ------------------------------------------------------------
# 3. Warm start 集合
# ------------------------------------------------------------
def warm_starts(rng):
    starts = []

    # (a) 多个随机排列
    for _ in range(8):
        starts.append(list(rng.permutation(N) + 1))

    # (b) NN by 距离（贪心走最近未访问）
    unv = set(range(1, N + 1)); cur = 0; route = []
    while unv:
        nxt = min(unv, key=lambda j: T[cur, j])
        route.append(nxt); unv.discard(nxt); cur = nxt
    starts.append(route)

    # (c) NN by 综合分（距离 + 时间窗紧迫度）
    unv = set(range(1, N + 1)); cur = 0; t = 0; route = []
    while unv:
        def score(j):
            arr = t + T[cur, j]
            late = max(0.0, arr - B[j])
            early = max(0.0, A[j] - arr)
            return T[cur, j] + 0.6 * late + 0.2 * early
        nxt = min(unv, key=score)
        route.append(nxt); t += T[cur, nxt] + S[nxt]; cur = nxt; unv.discard(nxt)
    starts.append(route)

    # (d) 按 a_i 升序
    starts.append(sorted(range(1, N + 1), key=lambda i: A[i]))
    # (e) 按 b_i 升序
    starts.append(sorted(range(1, N + 1), key=lambda i: B[i]))
    # (f) 按 (a_i+b_i)/2 升序
    starts.append(sorted(range(1, N + 1), key=lambda i: (A[i] + B[i]) / 2))
    return starts


# ------------------------------------------------------------
# 4. 局部搜索（基于 J）
# ------------------------------------------------------------
def two_opt_J(perm):
    """2-opt：反转任意子段，按 J 改进。"""
    n = len(perm)
    best = list(perm)
    best_J = evaluate(best)[2]
    iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True; iters += 1
    return best, best_J, iters


def or_opt_J(perm):
    """Or-opt：把长度 1/2/3 的子链整段迁移到其它位置。"""
    n = len(perm)
    best = list(perm)
    best_J = evaluate(best)[2]
    iters = 0
    improved = True
    while improved:
        improved = False
        for L in (1, 2, 3):
            for i in range(0, n - L + 1):
                seg = best[i:i + L]
                base = best[:i] + best[i + L:]
                for j in range(0, len(base) + 1):
                    if j == i:
                        continue
                    cand = base[:j] + seg + base[j:]
                    Jc = evaluate(cand)[2]
                    if Jc < best_J - 1e-9:
                        best = cand; best_J = Jc; improved = True; iters += 1
                        break
                if improved:
                    break
            if improved:
                break
    return best, best_J, iters


def swap_J(perm):
    """两两交换。"""
    n = len(perm)
    best = list(perm)
    best_J = evaluate(best)[2]
    iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True; iters += 1
    return best, best_J, iters


def polish(perm):
    """2-opt → Or-opt → swap 反复轮转直到 3 个都不再改进。"""
    cur = list(perm)
    cur_J = evaluate(cur)[2]
    total = 0
    while True:
        cur, J1, k1 = two_opt_J(cur)
        cur, J2, k2 = or_opt_J(cur)
        cur, J3, k3 = swap_J(cur)
        total += k1 + k2 + k3
        if k1 + k2 + k3 == 0:
            return cur, cur_J if k1 + k2 + k3 == 0 else J3, total
        cur_J = J3


# ------------------------------------------------------------
# 5. SA on permutation（邻域：swap / insert-segment / reverse）
# ------------------------------------------------------------
def sa_perm(perm0, rng, T0=80.0, Tf=1e-3, alpha=0.995, n_iter_per_T=200):
    cur = list(perm0)
    cur_J = evaluate(cur)[2]
    best = list(cur); best_J = cur_J
    history = [best_J]
    T_now = T0
    while T_now > Tf:
        for _ in range(n_iter_per_T):
            move = rng.choice(["swap", "insert", "reverse"])
            cand = cur.copy()
            n = len(cand)
            if move == "swap":
                i, j = rng.integers(0, n, size=2)
                if i == j:
                    continue
                cand[i], cand[j] = cand[j], cand[i]
            elif move == "insert":
                L = int(rng.integers(1, 4))
                if L >= n:
                    continue
                i = int(rng.integers(0, n - L + 1))
                seg = cand[i:i + L]
                base = cand[:i] + cand[i + L:]
                j = int(rng.integers(0, len(base) + 1))
                if j == i:
                    continue
                cand = base[:j] + seg + base[j:]
            else:  # reverse
                i, j = sorted(rng.integers(0, n, size=2).tolist())
                if i == j:
                    continue
                cand = cand[:i] + cand[i:j + 1][::-1] + cand[j + 1:]
            Jc = evaluate(cand)[2]
            dJ = Jc - cur_J
            if dJ < 0 or rng.random() < np.exp(-dJ / T_now):
                cur = cand; cur_J = Jc
                if cur_J < best_J:
                    best = list(cur); best_J = cur_J
        history.append(best_J)
        T_now *= alpha
    return best, best_J, history


# ------------------------------------------------------------
# 6. 主流程
# ------------------------------------------------------------
def main():
    print(f"== 问题 2 ==  n={N}  M1={M1}  M2={M2}  SEED={SEED}")
    rng = np.random.default_rng(SEED)

    starts = warm_starts(rng)
    print(f"[Warm starts] 共 {len(starts)} 个")

    # 6.1 多起点 polish
    t0 = time.time()
    polished = []
    for s in starts:
        p, J, _ = polish(s)
        polished.append((J, p))
    polished.sort(key=lambda kv: kv[0])
    best_polish_J, best_polish = polished[0]
    t_polish = time.time() - t0
    print(f"[Multi-start polish] {t_polish:.2f}s  best J={best_polish_J:.2f}")
    print(f"  TOP 3 J: {[round(kv[0], 2) for kv in polished[:3]]}")

    # 6.2 SA：以 polish 最优作初值，再做一遍长程退火
    t0 = time.time()
    sa_best, sa_J, sa_hist = sa_perm(best_polish, rng,
                                     T0=80.0, Tf=1e-3, alpha=0.995, n_iter_per_T=400)
    sa_polished, sa_polished_J, _ = polish(sa_best)
    t_sa = time.time() - t0
    print(f"[SA + polish] {t_sa:.2f}s  J(SA)={sa_J:.2f}  J(SA+polish)={sa_polished_J:.2f}")

    # 6.3 选 best
    if sa_polished_J < best_polish_J - 1e-9:
        final_perm, final_J = sa_polished, sa_polished_J
        source = "Multi-start polish + SA + polish"
    else:
        final_perm, final_J = best_polish, best_polish_J
        source = "Multi-start polish (SA 未改进)"

    travel, penalty, J, schedule = evaluate(final_perm, with_detail=True)
    full_route = [0] + list(final_perm) + [0]

    print(f"\n========== 问题 2 最终结果 ==========")
    print(f"  来源：{source}")
    print(f"  路径：{' -> '.join(map(str, full_route))}")
    print(f"  总运输时间 = {travel:.0f}")
    print(f"  时间窗惩罚 = {penalty:.2f}")
    print(f"  目标 J     = {J:.2f}")
    print(f"  违反客户：")
    for r in schedule:
        if r["early"] > 0 or r["late"] > 0:
            print(f"    客户 {r['customer']:>2}  到达={r['arrive']:>4.0f}  "
                  f"窗=[{r['tw_a']:.0f},{r['tw_b']:.0f}]  "
                  f"早到={r['early']:.0f} 晚到={r['late']:.0f}  惩罚={r['penalty']:.0f}")

    # 6.4 落盘 JSON
    result = dict(
        problem="Q2: 单车辆 + 时间窗惩罚（无容量），纯 Python 求解",
        n_customers=N,
        M1=M1, M2=M2,
        method=source,
        seeds=[SEED],
        time_sec=dict(polish=round(t_polish, 3), sa=round(t_sa, 3)),
        final=dict(
            route=full_route,
            perm=[int(x) for x in final_perm],
            total_travel_time=float(travel),
            total_tw_penalty=float(penalty),
            objective_J=float(J),
        ),
        schedule=schedule,
        topk_polish_J=[round(kv[0], 2) for kv in polished[:5]],
    )
    out_json = OUT_RESULT / "q2_pure_python.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}")

    # 6.5 表
    df = pd.DataFrame(schedule)
    df_csv = OUT_TABLE / "tab_02_q2_schedule.csv"
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{问题 2 单车辆调度（含时间窗违反）}\\label{tab:q2_schedule}\n"
        "\\begin{tabular}{ccccccc}\n\\toprule\n"
        "客户 & 到达时刻 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开时刻 \\\\\n\\midrule\n"
    )
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
    (OUT_TABLE / "tab_02_q2_schedule.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] {df_csv.relative_to(ROOT)} + .tex")

    # 6.6 路径图 + 时间窗甘特
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # (a) 环形路径
    ax = axes[0]
    K = len(full_route) - 1
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    xs, ys = np.cos(angles), np.sin(angles)
    for k in range(K):
        nxt = (k + 1) % K
        ax.annotate("", xy=(xs[nxt], ys[nxt]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color="#3a78c2", lw=1.4))
    for k in range(K):
        nid = full_route[k]
        # 标颜色：违反 = 红，正常 = 黄，depot = 灰
        color = "#888"
        if nid != 0:
            r = next(rr for rr in schedule if rr["customer"] == nid)
            if r["early"] > 0 or r["late"] > 0:
                color = "#d9534f"
            else:
                color = "#f0ad4e"
        ax.scatter(xs[k], ys[k], s=380, c=color, edgecolor="black", zorder=3)
        ax.text(xs[k], ys[k], str(nid), ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"问题 2 路径（红=时间窗违反, 橙=正常, 灰=depot）\n"
                 f"travel={travel:.0f}, penalty={penalty:.0f}, J={J:.0f}")

    # (b) 时间窗甘特
    ax = axes[1]
    yticks, ylabels = [], []
    for k, r in enumerate(schedule):
        y = k
        # 时间窗背景
        ax.barh(y, r["tw_b"] - r["tw_a"], left=r["tw_a"],
                height=0.5, color="#cfe5ff", edgecolor="#3a78c2", linewidth=0.8)
        # 实际服务区段
        sc = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else "#5cb85c"
        ax.barh(y, r["service"], left=r["arrive"], height=0.32, color=sc, alpha=0.95)
        ax.plot([r["arrive"], r["arrive"]], [y - 0.32, y + 0.32], color="black", lw=1.2)
        yticks.append(y); ylabels.append(f"客户 {r['customer']}")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("时间")
    ax.set_title("时间窗 vs 实际到达（绿=未违反, 红=违反）")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("问题 2：含时间窗惩罚的单车辆调度（纯 Python 多起点 + SA）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_02_q2_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_02_q2_route.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {(OUT_FIG / 'fig_02_q2_route.png').relative_to(ROOT)} (+ .pdf)")

    # 6.7 SA 收敛曲线
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(sa_hist, color="#5cb85c", lw=1.4)
    ax2.set_xlabel("外层降温步")
    ax2.set_ylabel("最佳 J（含惩罚）")
    ax2.set_title("问题 2 · SA 求解收敛曲线（外层每步内做 400 次邻域接受）")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_FIG / "fig_02_q2_convergence.png", dpi=300, bbox_inches="tight")
    fig2.savefig(OUT_FIG / "fig_02_q2_convergence.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"[写出] {(OUT_FIG / 'fig_02_q2_convergence.png').relative_to(ROOT)} (+ .pdf)")


if __name__ == "__main__":
    main()
