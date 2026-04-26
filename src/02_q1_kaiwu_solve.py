"""
问题 1：用 Kaiwu SDK 经典模拟退火求解器对 QUBO 进行 SDK 验证

仅做 SDK 端验证（本地经典 SA / 禁忌搜索 / 暴力搜索），与真机 CIM 解耦。
独立运行：不依赖 01_q1_qubo_tsp.py 的 Python 自实现 SA 结果，
仅调用 _q_lib 的 QUBO 构造与解码工具。

环境：D:/Anaconda/envs/kaiwu-py310/python.exe   (Python 3.10)
运行：D:/Anaconda/envs/kaiwu-py310/python.exe src/02_q1_kaiwu_solve.py

流程：
  1. kw.license.init(user_id, sdk_code)  鉴权
  2. _q_lib.build_qubo_q1()              得到 (225, 225) 上三角 QUBO
  3. kw.conversion.qubo_matrix_to_ising_matrix()  QUBO → Ising
  4. kw.classical.SimulatedAnnealingOptimizer     SDK 经典 SA 求解
  5. spin → binary，kw.qubo.calculate_qubo_value 复算 Q 值
  6. _q_lib.decode + route_cost 还原路径并校核
  7. 落盘 results/基础模型/qubo_v1_q1_kaiwu_sdk.json
       figures/fig_01_q1_sdk_hamiltonian.png（SDK SA 哈密顿量演化曲线）
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 仅在本进程内临时清除代理变量，不修改用户系统代理（kaiwu 服务在国内，避免被 SOCKS 拦截）
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
OUT_RESULT = ROOT / "results/基础模型"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
N = 15
A_PEN = 200.0
SEED = 20260426

# Kaiwu SDK 鉴权信息
KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"

# SA 求解器超参（SDK 经典 SA · 基线 baseline）
SA_INITIAL_T = 100.0
SA_ALPHA = 0.99
SA_CUTOFF_T = 1e-3
SA_ITERS_PER_T = 10
SA_SIZE_LIMIT = 50
SA_PROCESS_NUM = 1

# SA 求解器超参（SDK 经典 SA · 加强版 tuned）
# 设计理由（论文里要写）：
#   罚系数 A=200 ⇒ 单违约能量梯度 = 2A = 400
#   要让初始接受率 P=exp(-2A/T0) ≈ 0.45，需 T0 ≈ 500
#   alpha=0.995 ⇒ 退到 1e-3 共 ~2602 档（基线只有 ~1145 档），冷却更充分
#   iters_per_t=80 ⇒ 单链总翻转 ~208,160；分摊到 225 变量 ≈ 925 翻转/变量
#   size_limit=200 ⇒ 单次 solve 内并行 200 条独立链，去重后留更多样本
SA_TUNED = dict(
    initial_temperature=500.0,
    alpha=0.995,
    cutoff_temperature=1e-3,
    iterations_per_t=80,
    size_limit=200,
    process_num=1,
)
N_SEEDS = 5             # 加强版用 5 个不同种子集成（200×5 = 1000 个候选解）


# ------------------------------------------------------------
# 工具：spin (-1/+1) → binary (0/1)，并按行复算 QUBO 值
# ------------------------------------------------------------
def spin_to_binary(spin_solutions: np.ndarray, nvar: int) -> np.ndarray:
    """SDK Ising 求解器返回 shape=(k, nvar+1) 的 spin 解，最后一列是辅助 spin。
    用辅助 spin 做整体翻转归一化后，丢掉辅助维并转成 0/1。
    """
    s = np.asarray(spin_solutions, dtype=np.int8)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]                       # (k, 1) ∈ {-1, +1}
        s_fixed = s * s_aux                     # 让辅助 spin 都置成 +1
        s_main = s_fixed[:, :-1]                # 丢掉辅助维
    elif s.shape[1] == nvar:
        s_main = s
    else:
        raise ValueError(f"unexpected spin shape {s.shape}, expect (*, {nvar}) or (*, {nvar+1})")
    return ((s_main + 1) // 2).astype(np.int8)  # -1→0, +1→1


def evaluate_solutions(spin_solutions: np.ndarray,
                       Q_full: np.ndarray,
                       offset: float,
                       T: np.ndarray,
                       n: int):
    """对 SDK 返回的多解集合逐行评估。
    Args:
        spin_solutions: (k, nvar+1) ∈ {-1, +1}（含辅助 spin）
        Q_full: (nvar, nvar) 对称 QUBO 矩阵
        offset: QUBO ↔ Ising 转换的常数偏置
        T, n: 解码后路径代价复算所需
    Returns:
        list[dict] 按可行优先 + QUBO 值升序
    """
    nvar = Q_full.shape[0]
    x_all = spin_to_binary(spin_solutions, nvar)
    records = []
    for k in range(x_all.shape[0]):
        x = x_all[k]
        q_val = float(kw.qubo.calculate_qubo_value(Q_full, offset, x))
        perm, feasible = decode(x, n)
        cost = route_cost(perm, T) if feasible else float("inf")
        records.append(dict(
            x=x, qubo_value=q_val,
            perm=perm, feasible=feasible, cost=cost,
        ))
    records.sort(key=lambda r: (not r["feasible"], r["qubo_value"]))
    return records


# ------------------------------------------------------------
# 邻近邻 (NN) 启发式构造一个初始可行解（用作 SA 热启动）
# ------------------------------------------------------------
def nearest_neighbor_perm(T: np.ndarray, n: int, start_seed: int = 0) -> list[int]:
    """从 depot 出发的最近邻路径构造（带随机起点扰动以让多种子有差异）。"""
    rng = np.random.default_rng(start_seed)
    unvisited = set(range(1, n + 1))
    # 从 depot 出发，第一步随机在前 3 近的客户中挑一个
    cur = 0
    perm = []
    first_candidates = sorted(unvisited, key=lambda j: T[cur, j])[:3]
    nxt = int(rng.choice(first_candidates))
    perm.append(nxt)
    unvisited.remove(nxt)
    cur = nxt
    while unvisited:
        nxt = min(unvisited, key=lambda j: T[cur, j])
        perm.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return perm


def perm_to_binary(perm: list[int], n: int) -> np.ndarray:
    """长度 n 的客户排列 → one-hot binary 向量 (n*n,)，x[(i-1)*n + (p-1)] = 1 表示客户 i 在第 p 位"""
    x = np.zeros(n * n, dtype=np.int8)
    for p, i in enumerate(perm, start=1):
        x[(i - 1) * n + (p - 1)] = 1
    return x


def binary_to_spin_with_aux(x: np.ndarray) -> np.ndarray:
    """0/1 → spin (-1/+1)，并在末尾追加辅助 spin = +1。返回 shape=(nvar+1,)"""
    spin = (2 * x.astype(np.int8) - 1).astype(np.int8)
    return np.concatenate([spin, np.array([1], dtype=np.int8)])


# ------------------------------------------------------------
# 加强版求解：多种子 + 热启动 + 调优 SA 超参
# ------------------------------------------------------------
def tuned_solve(ising_mat, Q_full, ising_bias, T_mat, n, nvar):
    all_records = []
    per_seed_summary = []
    t0 = time.time()
    for s_idx in range(N_SEEDS):
        seed = SEED + s_idx
        # 给每个种子一个不同的近似可行热启动
        perm0 = nearest_neighbor_perm(T_mat, n, start_seed=seed)
        x0 = perm_to_binary(perm0, n)
        spin0_full = binary_to_spin_with_aux(x0)

        sa = kw.classical.SimulatedAnnealingOptimizer(
            **SA_TUNED,
            flag_evolution_history=False,   # 多种子时关掉省时
            verbose=False,
            rand_seed=seed,
        )
        spins = sa.solve(ising_mat, init_solution=spin0_full)
        if spins.ndim == 1:
            spins = spins.reshape(1, -1)
        recs = evaluate_solutions(spins, Q_full, ising_bias, T_mat, n)
        feas = [r for r in recs if r["feasible"]]
        per_seed_summary.append(dict(
            seed=seed,
            init_perm=perm0,
            init_cost=route_cost(perm0, T_mat),
            n_unique=int(spins.shape[0]),
            n_feasible=len(feas),
            best_cost=feas[0]["cost"] if feas else None,
            best_qubo=feas[0]["qubo_value"] if feas else recs[0]["qubo_value"],
        ))
        all_records.extend(recs)
    t_total = time.time() - t0
    all_records.sort(key=lambda r: (not r["feasible"], r["qubo_value"]))
    return all_records, per_seed_summary, t_total


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    print(f"== 问题 1 · Kaiwu SDK 验证 ==  n = {N}  A = {A_PEN}")

    # ---- 0) 鉴权 ----
    print("\n[1/5] license 初始化 ...")
    kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
    print(f"      user_id = {KAIWU_USER_ID}")
    print(f"      kaiwu   = {kw.__version__}")

    # ---- 1) 构造 QUBO ----
    print("\n[2/5] 构造 QUBO ...")
    T_mat, _ = load_data(N)
    Q_upper = build_qubo_q1(T_mat, N, A=A_PEN)         # 上三角形式
    Q_full = to_symmetric(Q_upper)                     # 对称形式（calculate_qubo_value 用）
    nvar = Q_upper.shape[0]
    const_term = 2 * N * A_PEN
    print(f"      变量数 nvar = {nvar}   (n^2 = {N*N})")
    print(f"      QUBO 非零项 (上三角) = {int(np.count_nonzero(Q_upper))}")

    # ---- 2) QUBO → Ising ----
    print("\n[3/5] QUBO → Ising 矩阵转换 ...")
    ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_upper)
    print(f"      ising_mat.shape = {ising_mat.shape}")
    print(f"      ising_bias      = {ising_bias:.4f}")

    # ---- 3) SDK 经典 SA 求解 ----
    print("\n[4/5] Kaiwu SDK · 经典模拟退火求解 ...")
    sa = kw.classical.SimulatedAnnealingOptimizer(
        initial_temperature=SA_INITIAL_T,
        alpha=SA_ALPHA,
        cutoff_temperature=SA_CUTOFF_T,
        iterations_per_t=SA_ITERS_PER_T,
        size_limit=SA_SIZE_LIMIT,
        flag_evolution_history=True,
        verbose=False,
        rand_seed=SEED,
        process_num=SA_PROCESS_NUM,
    )
    t0 = time.time()
    spin_sols = sa.solve(ising_mat)                    # 形状 (k, nvar)，k ≤ size_limit
    t_sa = time.time() - t0
    if spin_sols.ndim == 1:
        spin_sols = spin_sols.reshape(1, -1)
    print(f"      solve 耗时 = {t_sa:.2f}s   返回去重解数 = {spin_sols.shape[0]}")

    # ---- 4) 解码 + 评估 ----
    print("\n[5/5] 解码 / 复算 QUBO 值 / 计算路径代价 ...")
    records = evaluate_solutions(spin_sols, Q_full, ising_bias, T_mat, N)
    feas_recs = [r for r in records if r["feasible"]]
    print(f"      可行解数 = {len(feas_recs)} / {len(records)}")
    if feas_recs:
        best = feas_recs[0]
    else:
        best = records[0]
    print(f"      最优 QUBO 值 (含 offset) = {best['qubo_value']:.2f}")
    print(f"      const_term (理论可行解 QUBO 偏置) = {const_term:.2f}")
    print(f"      可行性 = {best['feasible']}   路径代价 = {best['cost']:.0f}")

    # 2-opt / Or-opt 打磨（仅作验证，不改 SDK 解本身）
    if best["feasible"]:
        polished, n_iters = hybrid_polish(best["perm"], T_mat)
        polished_cost = route_cost(polished, T_mat)
        polished_route = [0] + polished + [0]
        print(f"      + 2-opt/Or-opt 打磨: iters = {n_iters}  cost = {polished_cost:.0f}")
        print(f"      路径: {' -> '.join(map(str, polished_route))}")
    else:
        polished, polished_cost, n_iters = best["perm"], best["cost"], 0
        polished_route = [0] + polished + [0]
        print("      [警告] SDK 最优样本不满足 one-hot 约束，未打磨。")

    # ---- 5) 哈密顿量演化曲线 ----
    fig_path = None
    try:
        ha_hist = sa.get_ha_history()
        if ha_hist:
            xs = sorted(ha_hist.keys())
            ys = [ha_hist[k] for k in xs]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(xs, ys, lw=1.5, color="#1f77b4")
            ax.set_xlabel("时间 / s")
            ax.set_ylabel("哈密顿量 H (Ising 能量)")
            ax.set_title("Kaiwu SDK 经典 SA · 哈密顿量随退火演化")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig_path_png = OUT_FIG / "fig_01_q1_sdk_hamiltonian.png"
            fig_path_pdf = OUT_FIG / "fig_01_q1_sdk_hamiltonian.pdf"
            fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
            fig.savefig(fig_path_pdf, bbox_inches="tight")
            plt.close(fig)
            fig_path = str(fig_path_png.relative_to(ROOT))
            print(f"\n[写出] {fig_path}  (+ .pdf)")
    except Exception as e:
        print(f"\n[提示] 未能输出哈密顿量曲线: {e}")

    # ---- 6) 加强版求解：多种子 + 热启动 + 调优 SA 超参 ----
    print("\n[加强版] 多种子 + NN 热启动 + 调优 SA 超参 ...")
    print(f"           种子数 = {N_SEEDS}   单种子超参 = {SA_TUNED}")
    tuned_records, per_seed, t_tuned = tuned_solve(
        ising_mat, Q_full, ising_bias, T_mat, N, nvar
    )
    tuned_feas = [r for r in tuned_records if r["feasible"]]
    print(f"           总耗时 = {t_tuned:.2f}s   候选解总数 = {len(tuned_records)}")
    print(f"           可行解数 = {len(tuned_feas)} / {len(tuned_records)}  "
          f"(可行率 {len(tuned_feas)/max(len(tuned_records),1)*100:.1f}%)")
    if tuned_feas:
        tuned_best = tuned_feas[0]
        tuned_polished, tuned_iters = hybrid_polish(tuned_best["perm"], T_mat)
        tuned_polished_cost = route_cost(tuned_polished, T_mat)
        tuned_polished_route = [0] + tuned_polished + [0]
        print(f"           最优可行解 cost = {tuned_best['cost']:.0f}  "
              f"(QUBO = {tuned_best['qubo_value']:.1f})")
        print(f"           + 2-opt/Or-opt 打磨: iters = {tuned_iters}  "
              f"cost = {tuned_polished_cost:.0f}")
        print(f"           路径: {' -> '.join(map(str, tuned_polished_route))}")
    else:
        tuned_best = tuned_records[0]
        tuned_polished_cost, tuned_iters = float("inf"), 0
        tuned_polished_route = [0] + tuned_best["perm"] + [0]
        print("           [警告] 加强版仍未找到可行解。")

    # 各种子分项汇总
    print("\n           [各种子分项]")
    print("           seed       init_cost  unique  feas/total  best_cost")
    for s in per_seed:
        print(f"           {s['seed']:<10} {s['init_cost']:>9.0f}  "
              f"{s['n_unique']:>6}  {s['n_feasible']:>4}/{s['n_unique']:<5} "
              f"{(str(int(s['best_cost'])) if s['best_cost'] is not None else '-'):>9}")

    # ---- 7) 落盘 ----
    out = dict(
        problem="Q1 · Kaiwu SDK 验证（经典模拟退火求解 QUBO，独立于 Python 自实现 SA）",
        kaiwu_version=kw.__version__,
        n_customers=N,
        n_qubo_vars=nvar,
        penalty_A=A_PEN,
        ising_bias=float(ising_bias),
        const_term=const_term,
        sa_baseline_params=dict(
            initial_temperature=SA_INITIAL_T,
            alpha=SA_ALPHA,
            cutoff_temperature=SA_CUTOFF_T,
            iterations_per_t=SA_ITERS_PER_T,
            size_limit=SA_SIZE_LIMIT,
            rand_seed=SEED,
            process_num=SA_PROCESS_NUM,
        ),
        sdk_sa_baseline=dict(
            time_sec=round(t_sa, 3),
            n_unique_solutions=int(spin_sols.shape[0]),
            n_feasible=len(feas_recs),
            best_qubo_value=best["qubo_value"],
            best_feasible=best["feasible"],
            best_route_cost=best["cost"],
            best_perm=best["perm"],
        ),
        sdk_sa_baseline_plus_2opt=dict(
            two_opt_iters=int(n_iters),
            cost=polished_cost,
            route=polished_route,
        ),
        sa_tuned_params=dict(**SA_TUNED, n_seeds=N_SEEDS, base_seed=SEED),
        sdk_sa_tuned=dict(
            time_sec=round(t_tuned, 3),
            n_total_candidates=len(tuned_records),
            n_feasible=len(tuned_feas),
            feasibility_rate=round(len(tuned_feas)/max(len(tuned_records),1), 4),
            best_qubo_value=tuned_best["qubo_value"],
            best_feasible=tuned_best["feasible"],
            best_route_cost=tuned_best["cost"] if tuned_best["feasible"] else None,
            best_perm=tuned_best["perm"] if tuned_best["feasible"] else None,
            per_seed=[
                dict(seed=s["seed"], init_perm=s["init_perm"], init_cost=s["init_cost"],
                     n_unique=s["n_unique"], n_feasible=s["n_feasible"],
                     best_cost=s["best_cost"], best_qubo=s["best_qubo"])
                for s in per_seed
            ],
        ),
        sdk_sa_tuned_plus_2opt=dict(
            two_opt_iters=int(tuned_iters),
            cost=tuned_polished_cost if tuned_polished_cost != float("inf") else None,
            route=tuned_polished_route,
        ),
        figure=fig_path,
    )
    out_json = OUT_RESULT / "qubo_v1_q1_kaiwu_sdk.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else o),
                        encoding="utf-8")
    print(f"[写出] {out_json.relative_to(ROOT)}")
    print("\n== 完成 · Kaiwu SDK 验证 ==")


if __name__ == "__main__":
    main()
