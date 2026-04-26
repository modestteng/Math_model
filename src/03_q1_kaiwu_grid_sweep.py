"""
问题 1 · Kaiwu SDK 灵敏度分析：罚系数 A × 初始温度 T0 网格扫描

目的：定位"可行率 + 路径代价"的帕累托甜蜜区，回答论文中
   "为什么 A=200, T0=100 的默认参数可行率仅 2%"
   "调参后 SDK SA 能否在不依赖 2-opt 的情况下直接拿到优解"

设计：
  A   ∈ {50, 100, 150, 200, 300}     罚系数
  T0  ∈ {20, 50, 100, 200, 500}      初始温度
  共 5×5 = 25 个组合，每组 3 个种子（节省时间）
  每次 solve: alpha=0.995, cutoff=1e-3, iters_per_t=80, size_limit=200
  ⇒ 单组 600 个候选解，全表 15000 个候选解

输出：
  results/灵敏度分析/sens_A_T0_grid.json    全量数据
  results/灵敏度分析/sens_A_T0_grid.csv     扁平表
  figures/fig_q1_sens_feasibility_heatmap.png    可行率热力图
  figures/fig_q1_sens_bestcost_heatmap.png       最佳代价热力图
  tables/tab_q1_sens_A_T0.tex                    LaTeX 三线表

环境：D:/Anaconda/envs/kaiwu-py310/python.exe
运行：D:/Anaconda/envs/kaiwu-py310/python.exe src/03_q1_kaiwu_grid_sweep.py
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 仅在本进程内临时清除代理变量，不修改用户系统代理（kaiwu 国内服务）
for _v in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import kaiwu as kw  # type: ignore

from _q_lib import (
    load_data,
    build_qubo_q1,
    to_symmetric,
    decode,
    route_cost,
    hybrid_polish,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_RESULT = ROOT / "results/灵敏度分析"
OUT_FIG = ROOT / "figures"
OUT_TABLE = ROOT / "tables"
for p in (OUT_RESULT, OUT_FIG, OUT_TABLE):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
N = 15
SEED = 20260426

KAIWU_USER_ID = "147749086823493634"
KAIWU_SDK_CODE = "uYk15QcjxMqAcOeuzHeZ6DwKFzUX0g"

A_GRID = [50.0, 100.0, 150.0, 200.0, 300.0]
T0_GRID = [20.0, 50.0, 100.0, 200.0, 500.0]
N_SEEDS = 3
SA_FIXED = dict(
    alpha=0.995,
    cutoff_temperature=1e-3,
    iterations_per_t=80,
    size_limit=200,
    process_num=1,
    flag_evolution_history=False,
    verbose=False,
)


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def spin_to_binary(spin_solutions: np.ndarray, nvar: int) -> np.ndarray:
    s = np.asarray(spin_solutions, dtype=np.int8)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]
        s_fixed = s * s_aux
        s_main = s_fixed[:, :-1]
    elif s.shape[1] == nvar:
        s_main = s
    else:
        raise ValueError(f"unexpected spin shape {s.shape}")
    return ((s_main + 1) // 2).astype(np.int8)


def evaluate(spins, Q_full, offset, T_mat, n):
    nvar = Q_full.shape[0]
    x_all = spin_to_binary(spins, nvar)
    feas_count = 0
    best_cost = None
    best_qubo = float("inf")
    best_perm = None
    for k in range(x_all.shape[0]):
        x = x_all[k]
        q_val = float(kw.qubo.calculate_qubo_value(Q_full, offset, x))
        perm, feasible = decode(x, n)
        if feasible:
            feas_count += 1
            cost = route_cost(perm, T_mat)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_qubo = q_val
                best_perm = perm
        else:
            if best_cost is None and q_val < best_qubo:
                best_qubo = q_val
    return feas_count, best_cost, best_qubo, best_perm


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    print(f"== 问题 1 · A × T0 网格灵敏度扫描 ==  n = {N}")
    print(f"   A 网格  = {A_GRID}")
    print(f"   T0 网格 = {T0_GRID}")
    print(f"   每格种子数 = {N_SEEDS}   单次 size_limit = {SA_FIXED['size_limit']}")
    total_runs = len(A_GRID) * len(T0_GRID) * N_SEEDS
    total_cand = total_runs * SA_FIXED["size_limit"]
    print(f"   总求解次数 = {total_runs}   总候选解 = {total_cand}")

    print("\n[license]")
    kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
    print(f"   kaiwu = {kw.__version__}   user_id = {KAIWU_USER_ID}")

    print("\n[load data + build T_mat]")
    T_mat, _ = load_data(N)

    rows = []                                  # 扁平 DataFrame 行
    feas_grid = np.zeros((len(A_GRID), len(T0_GRID)))     # 平均可行率
    cost_grid = np.full((len(A_GRID), len(T0_GRID)), np.nan)  # 最佳 cost
    polished_grid = np.full_like(cost_grid, np.nan)            # 打磨后最佳 cost

    t_start = time.time()
    for ia, A_pen in enumerate(A_GRID):
        Q_upper = build_qubo_q1(T_mat, N, A=A_pen)
        Q_full = to_symmetric(Q_upper)
        ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_upper)
        for it, T0 in enumerate(T0_GRID):
            cell_feas = []
            cell_best_cost = None
            cell_best_perm = None
            cell_time = 0.0
            for s_idx in range(N_SEEDS):
                seed = SEED + ia * 100 + it * 10 + s_idx
                sa = kw.classical.SimulatedAnnealingOptimizer(
                    initial_temperature=T0,
                    rand_seed=seed,
                    **SA_FIXED,
                )
                t0 = time.time()
                spins = sa.solve(ising_mat)
                dt = time.time() - t0
                cell_time += dt
                feas_count, best_cost, best_qubo, best_perm = evaluate(
                    spins, Q_full, ising_bias, T_mat, N
                )
                k = spins.shape[0] if spins.ndim > 1 else 1
                cell_feas.append(feas_count / k)
                if best_cost is not None and (cell_best_cost is None or best_cost < cell_best_cost):
                    cell_best_cost = best_cost
                    cell_best_perm = best_perm
                rows.append(dict(
                    A=A_pen, T0=T0, seed=seed,
                    n_unique=k, n_feasible=feas_count,
                    feas_rate=feas_count / k,
                    best_cost=best_cost,
                    best_qubo=best_qubo,
                    time_sec=round(dt, 3),
                ))
            feas_grid[ia, it] = float(np.mean(cell_feas))
            if cell_best_cost is not None:
                cost_grid[ia, it] = cell_best_cost
                polished_perm, _ = hybrid_polish(cell_best_perm, T_mat)
                polished_grid[ia, it] = route_cost(polished_perm, T_mat)
            print(f"   A={A_pen:>5.0f}  T0={T0:>5.0f}  "
                  f"feas_rate={feas_grid[ia, it]*100:>5.1f}%  "
                  f"best_cost={'-' if cell_best_cost is None else f'{cell_best_cost:>4.0f}'}  "
                  f"polished={'-' if np.isnan(polished_grid[ia, it]) else f'{polished_grid[ia, it]:>4.0f}'}  "
                  f"({cell_time:.2f}s)")
    t_total = time.time() - t_start
    print(f"\n[扫描总耗时] {t_total:.1f}s  =  {t_total/60:.2f} min")

    # ---- 落盘 JSON + CSV ----
    df = pd.DataFrame(rows)
    csv_path = OUT_RESULT / "sens_A_T0_grid.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[写出] {csv_path.relative_to(ROOT)}")

    out_json = dict(
        problem="Q1 · A × T0 灵敏度扫描（Kaiwu SDK 经典 SA）",
        kaiwu_version=kw.__version__,
        n_customers=N,
        A_grid=A_GRID,
        T0_grid=T0_GRID,
        n_seeds=N_SEEDS,
        sa_fixed=SA_FIXED,
        feas_rate_grid=feas_grid.tolist(),
        best_cost_grid=cost_grid.tolist(),
        polished_cost_grid=polished_grid.tolist(),
        runs=rows,
        total_time_sec=round(t_total, 1),
    )
    json_path = OUT_RESULT / "sens_A_T0_grid.json"
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2,
                                    default=lambda o: o.tolist() if hasattr(o, "tolist") else o),
                         encoding="utf-8")
    print(f"[写出] {json_path.relative_to(ROOT)}")

    # ---- 热力图 1：可行率 ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(feas_grid * 100, cmap="YlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(T0_GRID)))
    ax.set_xticklabels([f"{int(t)}" for t in T0_GRID])
    ax.set_yticks(range(len(A_GRID)))
    ax.set_yticklabels([f"{int(a)}" for a in A_GRID])
    ax.set_xlabel(r"初始温度 $T_0$")
    ax.set_ylabel(r"罚系数 $A$")
    ax.set_title(r"Kaiwu SDK 经典 SA 可行率 (%) 在 $A \times T_0$ 网格上的分布")
    for i in range(len(A_GRID)):
        for j in range(len(T0_GRID)):
            v = feas_grid[i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="black" if v < 50 else "white", fontsize=10)
    fig.colorbar(im, ax=ax, label="可行率 / %")
    fig.tight_layout()
    p1 = OUT_FIG / "fig_q1_sens_feasibility_heatmap.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {p1.relative_to(ROOT)}  (+ .pdf)")

    # ---- 热力图 2：最佳代价（含打磨）----
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, grid, title in zip(axs,
                                [cost_grid, polished_grid],
                                ["SDK SA 原始最优 cost", "SDK SA + 2-opt/Or-opt 打磨 cost"]):
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap="viridis_r", aspect="auto", origin="lower")
        ax.set_xticks(range(len(T0_GRID)))
        ax.set_xticklabels([f"{int(t)}" for t in T0_GRID])
        ax.set_yticks(range(len(A_GRID)))
        ax.set_yticklabels([f"{int(a)}" for a in A_GRID])
        ax.set_xlabel(r"初始温度 $T_0$")
        ax.set_ylabel(r"罚系数 $A$")
        ax.set_title(title)
        for i in range(len(A_GRID)):
            for j in range(len(T0_GRID)):
                v = grid[i, j]
                txt = "—" if np.isnan(v) else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white", fontsize=10)
        fig.colorbar(im, ax=ax, label="路径代价")
    fig.suptitle(r"$A \times T_0$ 网格上的最佳路径代价")
    fig.tight_layout()
    p2 = OUT_FIG / "fig_q1_sens_bestcost_heatmap.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    fig.savefig(p2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {p2.relative_to(ROOT)}  (+ .pdf)")

    # ---- LaTeX 三线表 ----
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"  \centering")
    tex.append(r"  \caption{问题 1 · Kaiwu SDK 经典 SA 在 $A \times T_0$ 网格上的可行率与最佳代价}")
    tex.append(r"  \label{tab:q1_sens_A_T0}")
    tex.append(r"  \begin{tabular}{cc" + "c" * len(T0_GRID) + r"}")
    tex.append(r"    \toprule")
    tex.append(r"    & $T_0=$ & " + " & ".join(f"{int(t)}" for t in T0_GRID) + r" \\")
    tex.append(r"    \midrule")
    for ia, A_pen in enumerate(A_GRID):
        cells = []
        for it in range(len(T0_GRID)):
            f = feas_grid[ia, it] * 100
            c = polished_grid[ia, it]
            cells.append(f"{f:.0f}\\%/{'-' if np.isnan(c) else f'{int(c)}'}")
        tex.append(f"    $A={int(A_pen)}$ & & " + " & ".join(cells) + r" \\")
    tex.append(r"    \bottomrule")
    tex.append(r"  \end{tabular}")
    tex.append(r"  \begin{flushleft}\footnotesize")
    tex.append(r"  注：每格 $a/b$ 表示\textbf{平均可行率 $a$\% / 打磨后最佳路径代价 $b$}；"
               r"`-' 表示三个种子均未找到可行解。每格 3 个种子 $\times$ 200 个候选解。")
    tex.append(r"  \end{flushleft}")
    tex.append(r"\end{table}")
    p3 = OUT_TABLE / "tab_q1_sens_A_T0.tex"
    p3.write_text("\n".join(tex), encoding="utf-8")
    print(f"[写出] {p3.relative_to(ROOT)}")

    # ---- 控制台总结 ----
    valid = ~np.isnan(polished_grid)
    if valid.any():
        i_best, j_best = np.unravel_index(np.nanargmin(polished_grid), polished_grid.shape)
        print(f"\n[最优组合] A = {A_GRID[i_best]}  T0 = {T0_GRID[j_best]}  "
              f"打磨后 cost = {polished_grid[i_best, j_best]:.0f}  "
              f"该格可行率 = {feas_grid[i_best, j_best]*100:.1f}%")
        i_h, j_h = np.unravel_index(np.argmax(feas_grid), feas_grid.shape)
        print(f"[最高可行率] A = {A_GRID[i_h]}  T0 = {T0_GRID[j_h]}  "
              f"可行率 = {feas_grid[i_h, j_h]*100:.1f}%  "
              f"该格打磨后 cost = "
              f"{'-' if np.isnan(polished_grid[i_h, j_h]) else f'{polished_grid[i_h, j_h]:.0f}'}")
    print("\n== 完成 · A × T0 灵敏度扫描 ==")


if __name__ == "__main__":
    main()
