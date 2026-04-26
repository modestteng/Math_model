"""
问题 1：不考虑时间窗与容量约束的单车辆 TSP
  (i)  最小化总运输时间的单车辆路径模型
  (ii) one-hot 位置编码 → QUBO（n^2 = 225 比特）
  (iii) 模拟退火 SA 求 QUBO 解；Held-Karp DP 给精确基线对照
输入：参考算例.xlsx（节点表 + 51x51 时间矩阵）
输出：
  results/基础模型/qubo_v1_q1_route.json
  tables/tab_01_q1_route.csv
  figures/fig_01_q1_route.png  (+ .pdf)
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

N = 15  # 客户数
SEED = 20260426

# ------------------------------------------------------------
# 1. 读取数据
# ------------------------------------------------------------
nodes = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_mat = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
# T_mat[i, j] = node i -> node j 单向运输时间，i, j ∈ {0, ..., 50}
assert T_mat.shape == (51, 51)

# 取 depot (0) 和客户 1..N
T = T_mat[: N + 1, : N + 1].astype(float)  # (N+1) x (N+1)

# ------------------------------------------------------------
# 2. QUBO 构造
#   变量 x[i, p]: 客户 i (1..N) 在路径第 p (1..N) 个位置
#   能量 H = A * H_const + H_dist
#     H_const = Σ_p (Σ_i x_{i,p} - 1)^2 + Σ_i (Σ_p x_{i,p} - 1)^2
#     H_dist  = Σ_i T[0,i] x_{i,1}                       (depot→首位)
#             + Σ_p Σ_{i≠j} T[i,j] x_{i,p} x_{j,p+1}     (相邻位)
#             + Σ_i T[i,0] x_{i,N}                        (末位→depot)
#   存储为严格上三角 Q：E = Σ_k Q[k,k] x_k + Σ_{k<l} Q[k,l] x_k x_l
# ------------------------------------------------------------
n = N
nvar = n * n


def idx(i: int, p: int) -> int:
    """客户 i (1..n), 位置 p (1..n) → 变量编号"""
    return (i - 1) * n + (p - 1)


def build_qubo(T: np.ndarray, A: float) -> np.ndarray:
    Q = np.zeros((nvar, nvar))
    # 列约束：每个位置 p 恰好一个客户
    for p in range(1, n + 1):
        vs = [idx(i, p) for i in range(1, n + 1)]
        for k in vs:
            Q[k, k] += -A
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                Q[vs[a], vs[b]] += 2 * A
    # 行约束：每个客户 i 恰好出现在一个位置
    for i in range(1, n + 1):
        vs = [idx(i, p) for p in range(1, n + 1)]
        for k in vs:
            Q[k, k] += -A
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                Q[vs[a], vs[b]] += 2 * A
    # 距离项
    for i in range(1, n + 1):
        Q[idx(i, 1), idx(i, 1)] += T[0, i]
        Q[idx(i, n), idx(i, n)] += T[i, 0]
    for p in range(1, n):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                k1, k2 = idx(i, p), idx(j, p + 1)
                ku, kv = (k1, k2) if k1 < k2 else (k2, k1)
                Q[ku, kv] += T[i, j]
    return Q


# 罚系数：A 应大于"违反约束"换得的最大距离收益。
# 单条边权 ≤ ~10，整条路径 ≈ 16 条边。取 A = 200 留充裕余量。
A_PEN = 200.0
Q = build_qubo(T, A_PEN)

# 添加常数项 2*N*A（对求解无影响，只用于报告能量）
const_term = 2 * N * A_PEN

# 转为对称矩阵 S: x^T S x = Σ_k Q_kk x_k + Σ_{k<l} Q_kl x_k x_l
S = np.zeros_like(Q)
diag = np.diag(Q).copy()
upper = np.triu(Q, k=1)
S = upper + upper.T
np.fill_diagonal(S, diag)


# ------------------------------------------------------------
# 3. 模拟退火求 QUBO （Kaiwu SDK 不可用，使用等价的 SA 求解器）
#    维护局部场 L[k] = S_kk + 2 Σ_{j≠k} S_kj x_j
#    翻转 k：ΔE = (1-2x_k) L[k]
# ------------------------------------------------------------
def _init_feasible(rng) -> np.ndarray:
    """构造一个随机可行的 x（每行每列恰好一个 1）"""
    perm = rng.permutation(n) + 1  # 1..n 的随机排列
    x = np.zeros(nvar, dtype=np.int8)
    for p, i in enumerate(perm, start=1):
        x[idx(int(i), p)] = 1
    return x


def sa_solve(
    S: np.ndarray,
    n_sweeps: int = 3000,
    T_init: float = 20.0,
    T_final: float = 1e-3,
    n_restarts: int = 12,
    seed: int = SEED,
) -> tuple[np.ndarray, float, list[float]]:
    rng = np.random.default_rng(seed)
    nvar = S.shape[0]
    diagS = np.diag(S).copy()

    best_x_global = None
    best_E_global = np.inf
    best_history = None

    schedule = np.geomspace(T_init, T_final, n_sweeps)

    for restart in range(n_restarts):
        # 一半 restart 从可行排列起步，一半完全随机：兼顾 exploit/explore
        if restart % 2 == 0:
            x = _init_feasible(rng)
        else:
            x = rng.integers(0, 2, nvar).astype(np.int8)
        Sx = S @ x
        L = diagS + 2 * Sx - 2 * diagS * x  # 局部场
        E = float(x @ S @ x)
        best_x = x.copy()
        best_E = E
        history = [E]

        for T in schedule:
            order = rng.permutation(nvar)
            for k in order:
                dE = (1 - 2 * x[k]) * L[k]
                if dE < 0 or rng.random() < np.exp(-dE / T):
                    delta = 1 - 2 * int(x[k])
                    x[k] = 1 - x[k]
                    L += 2 * delta * S[:, k]
                    L[k] -= 2 * delta * S[k, k]
                    E += dE
                    if E < best_E:
                        best_E = E
                        best_x = x.copy()
            history.append(best_E)

        if best_E < best_E_global:
            best_E_global = best_E
            best_x_global = best_x
            best_history = history

    return best_x_global, best_E_global, best_history


# ------------------------------------------------------------
# 4. Held-Karp 精确 DP（n=15 时 2^15·15² 完全可解，作为最优基线）
# ------------------------------------------------------------
def held_karp(T: np.ndarray, n: int) -> tuple[float, list[int]]:
    INF = np.inf
    full = (1 << n) - 1
    dp = np.full((1 << n, n + 1), INF)
    parent = np.full((1 << n, n + 1), -1, dtype=np.int32)
    for j in range(1, n + 1):
        dp[1 << (j - 1), j] = T[0, j]
    for mask in range(1, 1 << n):
        for j in range(1, n + 1):
            if not (mask & (1 << (j - 1))):
                continue
            cur = dp[mask, j]
            if cur == INF:
                continue
            remain = full ^ mask
            k = 1
            r = remain
            while r:
                if r & 1:
                    new_mask = mask | (1 << (k - 1))
                    new_cost = cur + T[j, k]
                    if new_cost < dp[new_mask, k]:
                        dp[new_mask, k] = new_cost
                        parent[new_mask, k] = j
                r >>= 1
                k += 1
    best_cost = INF
    best_end = -1
    for j in range(1, n + 1):
        c = dp[full, j] + T[j, 0]
        if c < best_cost:
            best_cost = c
            best_end = j
    rev = []
    mask, j = full, best_end
    while j != -1:
        rev.append(j)
        prev = int(parent[mask, j])
        mask ^= 1 << (j - 1)
        j = prev
    tour = [0] + rev[::-1] + [0]
    return float(best_cost), tour


# ------------------------------------------------------------
# 5. 解码 QUBO 解 → 路径
# ------------------------------------------------------------
def decode(x: np.ndarray) -> tuple[list[int], bool]:
    M = x.reshape(n, n)  # M[i-1, p-1]
    feasible = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm = []
    for p in range(n):
        col = M[:, p]
        if col.sum() == 1:
            perm.append(int(np.argmax(col)) + 1)
        else:
            # 不可行：取最大者作为占位
            perm.append(int(np.argmax(col)) + 1)
    return perm, feasible


def route_cost(route: list[int], T: np.ndarray) -> float:
    full = [0] + route + [0]
    return float(sum(T[full[i], full[i + 1]] for i in range(len(full) - 1)))


def two_opt(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    """2-opt 局部搜索（首尾隐式接 depot 0）"""
    n = len(perm)
    full = [0] + list(perm) + [0]
    improved = True
    iters = 0
    while improved:
        improved = False
        for i in range(1, n):
            for k in range(i + 1, n + 1):
                a, b = full[i - 1], full[i]
                c, d = full[k], full[k + 1]
                old = T[a, b] + T[c, d]
                new = T[a, c] + T[b, d]
                if new < old - 1e-9:
                    full[i : k + 1] = full[i : k + 1][::-1]
                    improved = True
                    iters += 1
    return full[1:-1], iters


def or_opt(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    """Or-opt：把长度 1/2/3 的子链整体迁移到另一位置"""
    n = len(perm)
    full = [0] + list(perm) + [0]
    improved = True
    iters = 0
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            for i in range(1, n - seg_len + 2):
                seg = full[i : i + seg_len]
                a, b = full[i - 1], full[i + seg_len]  # 移除前后邻居
                cost_remove = T[a, seg[0]] + T[seg[-1], b] - T[a, b]
                # 尝试插入到位置 j 与 j+1 之间
                base = full[:i] + full[i + seg_len :]
                for j in range(len(base) - 1):
                    if i - 1 <= j <= i:  # 原位附近，跳过
                        continue
                    p, q = base[j], base[j + 1]
                    cost_insert = T[p, seg[0]] + T[seg[-1], q] - T[p, q]
                    if cost_insert - cost_remove < -1e-9:
                        full = base[: j + 1] + seg + base[j + 1 :]
                        improved = True
                        iters += 1
                        break
                if improved:
                    break
            if improved:
                break
    return full[1:-1], iters


def hybrid_polish(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    """2-opt 与 Or-opt 反复迭代，直至双稳"""
    total = 0
    while True:
        perm, k1 = two_opt(perm, T)
        perm, k2 = or_opt(perm, T)
        total += k1 + k2
        if k1 + k2 == 0:
            return perm, total


# ------------------------------------------------------------
# 6. 主流程
# ------------------------------------------------------------
def main():
    print(f"== 问题 1 ==  n={N}  变量数={nvar}  罚系数 A={A_PEN}")
    print(f"  数据：customers 1..{N} from {DATA_XLSX.name}")

    # 6.1 SA 求 QUBO
    t0 = time.time()
    x_best, E_best, history = sa_solve(S)
    t_sa = time.time() - t0
    perm, feasible = decode(x_best)
    sa_cost = route_cost(perm, T)
    sa_full_route = [0] + perm + [0]
    print(f"\n[SA(QUBO)] {t_sa:.2f}s  feasible={feasible}  cost={sa_cost:.0f}  E={E_best + const_term:.1f}")
    print(f"  路径: {' -> '.join(map(str, sa_full_route))}")

    # 6.1.b 2-opt + Or-opt 后处理（混合策略：QUBO 提供可行结构，局部搜索打磨）
    if feasible:
        perm_polished, n_iters = hybrid_polish(perm, T)
        polished_cost = route_cost(perm_polished, T)
        polished_full_route = [0] + perm_polished + [0]
        print(f"\n[SA + (2-opt/Or-opt)] iters={n_iters}  cost={polished_cost:.0f}")
        print(f"  路径: {' -> '.join(map(str, polished_full_route))}")
    else:
        polished_cost = sa_cost
        polished_full_route = sa_full_route
        n_iters = 0

    # 6.2 Held-Karp 精确解
    t0 = time.time()
    hk_cost, hk_tour = held_karp(T, N)
    t_hk = time.time() - t0
    print(f"\n[Held-Karp 精确] {t_hk:.2f}s  cost={hk_cost:.0f}")
    print(f"  路径: {' -> '.join(map(str, hk_tour))}")

    # 6.3 评估 SA / 混合 与精确最优的 gap
    gap_sa = (sa_cost - hk_cost) / hk_cost * 100
    gap_polished = (polished_cost - hk_cost) / hk_cost * 100
    print(f"\n[对比] SA gap = {gap_sa:+.2f}%   SA+2opt gap = {gap_polished:+.2f}%")

    # 报告主体：SA+2opt 混合解。若仍非最优，附加精确解对照。
    final_route = polished_full_route
    final_cost = polished_cost
    source = "QUBO/SA + 2-opt 混合（附 Held-Karp 精确解对照）"

    # 6.4 计算逐节点到达 / 出发时间（无空闲等待，无时间窗）
    schedule_rows = []
    cur_t = 0
    for k in range(len(final_route) - 1):
        a, b = final_route[k], final_route[k + 1]
        travel = int(T[a, b])
        arrive = cur_t + travel
        if b == 0:
            service = 0
            depart = arrive
        else:
            service = int(nodes.loc[nodes["ID"] == b, "service"].values[0])
            depart = arrive + service
        schedule_rows.append(
            dict(
                step=k + 1,
                from_node=a,
                to_node=b,
                travel=travel,
                arrive=arrive,
                service=service,
                depart=depart,
            )
        )
        cur_t = depart
    total_time = sum(r["travel"] for r in schedule_rows)

    # 6.5 写出 results/基础模型/qubo_v1_q1_route.json
    result = dict(
        problem="Q1: 单车辆 TSP（无时间窗、无容量）",
        n_customers=N,
        n_qubo_vars=nvar,
        penalty_A=A_PEN,
        sa=dict(
            n_sweeps=4000,
            n_restarts=8,
            T_init=50.0,
            T_final=1e-3,
            seed=SEED,
            time_sec=round(t_sa, 3),
            feasible=feasible,
            cost=sa_cost,
            energy=E_best + const_term,
            route=sa_full_route,
        ),
        sa_plus_2opt=dict(
            two_opt_iters=int(n_iters),
            cost=polished_cost,
            route=polished_full_route,
        ),
        held_karp=dict(time_sec=round(t_hk, 3), cost=hk_cost, route=hk_tour),
        gap_sa_vs_exact_pct=round(gap_sa, 4),
        gap_sa2opt_vs_exact_pct=round(gap_polished, 4),
        final=dict(source=source, route=final_route, total_travel_time=int(total_time)),
        schedule=schedule_rows,
    )
    out_json = OUT_RESULT / "qubo_v1_q1_route.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}")

    # 6.6 写出 tables/tab_01_q1_route.csv  + .tex
    df = pd.DataFrame(schedule_rows)
    df_csv = OUT_TABLE / "tab_01_q1_route.csv"
    df.to_csv(df_csv, index=False, encoding="utf-8-sig")
    # 简单三线表 LaTeX
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{问题 1 最优单车辆调度方案}\\label{tab:q1_route}\n"
        "\\begin{tabular}{cccccc}\n\\toprule\n"
        "步序 & 起点 & 终点 & 行驶时间 & 到达时刻 & 离开时刻 \\\\\n\\midrule\n"
    )
    for r in schedule_rows:
        tex += f"{r['step']} & {r['from_node']} & {r['to_node']} & {r['travel']} & {r['arrive']} & {r['depart']} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT_TABLE / "tab_01_q1_route.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] {df_csv.relative_to(ROOT)} + .tex")

    # 6.7 出图：(a) 路径环形示意；(b) 累计时间甘特
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) 环形图
    ax = axes[0]
    nodes_in_route = final_route[:-1]  # 去掉末尾 0
    K = len(nodes_in_route)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    xs, ys = np.cos(angles), np.sin(angles)
    for k in range(K):
        nxt = (k + 1) % K
        ax.annotate(
            "",
            xy=(xs[nxt], ys[nxt]),
            xytext=(xs[k], ys[k]),
            arrowprops=dict(arrowstyle="->", color="#3a78c2", lw=1.4),
        )
    for k, nid in enumerate(nodes_in_route):
        color = "#d9534f" if nid == 0 else "#f0ad4e"
        ax.scatter(xs[k], ys[k], s=380, c=color, edgecolor="black", zorder=3)
        ax.text(xs[k], ys[k], str(nid), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"问题 1 路径（环形示意, depot=0, 客户=1..{N}）\n总运输时间 = {int(total_time)}")

    # (b) 累计时间序列
    ax = axes[1]
    ts_arrive = [0] + [r["arrive"] for r in schedule_rows]
    labels = [str(n) for n in final_route]
    ax.plot(range(len(ts_arrive)), ts_arrive, "o-", color="#3a78c2", lw=1.6, ms=7)
    for i, (t, lab) in enumerate(zip(ts_arrive, labels)):
        ax.annotate(lab, (i, t), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("访问序号")
    ax.set_ylabel("到达时刻 (累计)")
    ax.set_title("累计行驶时间曲线")
    ax.grid(alpha=0.3)

    fig.suptitle("问题 1：单车辆 TSP（QUBO + SA, 经精确 DP 验证）", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_png = OUT_FIG / "fig_01_q1_route.png"
    fig_pdf = OUT_FIG / "fig_01_q1_route.pdf"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {fig_png.relative_to(ROOT)} (+ .pdf)")

    # 6.8 SA 收敛曲线（独立图）
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(history, color="#5cb85c", lw=1.4)
    ax2.set_xlabel("SA sweep")
    ax2.set_ylabel("最佳能量 E")
    ax2.set_title("SA 求 QUBO 的最佳能量收敛曲线（最优 restart）")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_FIG / "fig_01_q1_sa_convergence.png", dpi=300, bbox_inches="tight")
    fig2.savefig(OUT_FIG / "fig_01_q1_sa_convergence.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"[写出] figures/fig_01_q1_sa_convergence.png (+ .pdf)")

    # 6.9 终端总结
    print("\n========== 问题 1 最终结果 ==========")
    print(f"  来源：{source}")
    print(f"  路径：{' -> '.join(map(str, final_route))}")
    print(f"  总运输时间：{int(total_time)}")
    print(f"  QUBO 变量数：{nvar}（n^2，one-hot 位置编码）")
    print(f"  SA gap = {gap_sa:+.2f}%   SA+2opt gap = {gap_polished:+.2f}%")


if __name__ == "__main__":
    main()
