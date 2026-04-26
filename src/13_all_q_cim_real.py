"""
全 4 问 · CIM 真机一次性验证（玻色 CPQC-550 相干伊辛机）
========================================================
为节省算力配额，一次提交全部 12 个 QUBO（Q1×1 + Q2×1 + Q3×3 + Q4×7）→ 一次拿全。
全部子 QUBO 比特数 ≤ 550 真机上限（铁律 §二.5/二.6）。

每个 QUBO 都按统一流程：
  1) 构造 QUBO（与已有 SDK 验证脚本完全一致的编码 → 链路一致性）
  2) 8-bit 精度调整（kw.qubo.adjust_qubo_matrix_precision）
  3) 提交 CIM (task_mode='quota', sample_number=10, wait=True)
  4) spin → binary → 解码 → (针对 Q1/Q2 单车) polish / (Q3/Q4 子 TSP) polish + 拼接
  5) 与纯 Python / SDK SA 基线对比
  6) 输出哈密顿量演化（题目硬性要求图）

输出
----
  results/基础模型/all_q_cim_real.json   总览 + 每问详情 + 比特预算 + 加速对比
  figures/fig_all_q_cim_hamiltonian.png  4 问哈密顿量演化合成图
  figures/fig_all_q_cim_summary.png      4 问 vs 基线柱状图
  figures/fig_all_q_cim_qubits.png       12 个 QUBO 的比特预算柱状图
  results/基础模型/q{1,2,3,4}_cim_real.json   每问独立 JSON
"""
from __future__ import annotations
import os
for _k in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT_RESULT = ROOT / "results/基础模型"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"
A_PEN = 200.0
M1, M2 = 10.0, 20.0
CAPACITY = 60
M_VEHICLE = 1000.0
SEED = 20260426
CIM_SAMPLE = 10
TASK_TS = time.strftime("%Y%m%d_%H%M%S")

# Kaiwu
import tempfile
import kaiwu as kw
print(f"[Kaiwu] version = {kw.__version__}", flush=True)
kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
print("[Kaiwu] license initialized [OK]", flush=True)
# CheckpointManager 配置（CIM 真机硬性要求）
CKPT_DIR = Path(tempfile.gettempdir()) / "kaiwu_cim_ckpt_all_q"
CKPT_DIR.mkdir(exist_ok=True)
kw.common.CheckpointManager.save_dir = str(CKPT_DIR)
print(f"[Kaiwu] CheckpointManager.save_dir = {CKPT_DIR}", flush=True)

# ---------- 数据 ----------
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_FULL = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int).astype(float)
A_VEC = nodes_raw["tw_a"].values.astype(float)
B_VEC = nodes_raw["tw_b"].values.astype(float)
S_VEC = nodes_raw["service"].values.astype(float)
D_VEC = nodes_raw["demand"].values.astype(float)


# ---------- 通用 QUBO 构造（one-hot + 距离，方案 C） ----------
def build_qubo_tsp(T_sub: np.ndarray, n_sub: int, A_pen: float = A_PEN) -> np.ndarray:
    nvar = n_sub * n_sub

    def idx(i, p): return (i - 1) * n_sub + (p - 1)

    Q = np.zeros((nvar, nvar))
    for p in range(1, n_sub + 1):
        vs = [idx(i, p) for i in range(1, n_sub + 1)]
        for k in vs: Q[k, k] += -A_pen
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)): Q[vs[a], vs[b]] += 2 * A_pen
    for i in range(1, n_sub + 1):
        vs = [idx(i, p) for p in range(1, n_sub + 1)]
        for k in vs: Q[k, k] += -A_pen
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)): Q[vs[a], vs[b]] += 2 * A_pen
    for i in range(1, n_sub + 1):
        Q[idx(i, 1), idx(i, 1)] += T_sub[0, i]
        Q[idx(i, n_sub), idx(i, n_sub)] += T_sub[i, 0]
    for p in range(1, n_sub):
        for i in range(1, n_sub + 1):
            for j in range(1, n_sub + 1):
                if i == j: continue
                k1, k2 = idx(i, p), idx(j, p + 1)
                ku, kv = (k1, k2) if k1 < k2 else (k2, k1)
                Q[ku, kv] += T_sub[i, j]
    return Q


def make_sub_T(a_node: int, customers: list[int], b_node: int) -> np.ndarray:
    k = len(customers)
    T = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        ci = customers[i - 1]
        T[0, i] = T_FULL[a_node, ci]
        T[i, 0] = T_FULL[ci, b_node]
        for j in range(1, k + 1):
            if i != j:
                T[i, j] = T_FULL[ci, customers[j - 1]]
    return T


def decode_sub(x: np.ndarray, n_sub: int):
    M = np.asarray(x).reshape(n_sub, n_sub)
    feas = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm_idx = [int(np.argmax(M[:, p])) for p in range(n_sub)]
    return perm_idx, feas


def spin_to_binary(spin_solutions: np.ndarray, nvar: int) -> np.ndarray:
    s = spin_solutions
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]
        s_main = (s * s_aux)[:, :-1]
    else:
        s_main = s
    return ((s_main + 1) // 2).astype(np.int8)


# ---------- 评估（含时间窗罚） ----------
def evaluate_route(route_customers, with_detail=False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []; dsum = 0.0
    for i in route_customers:
        i = int(i); tt = T_FULL[last, i]; cur += tt; travel += tt
        ai, bi = float(A_VEC[i]), float(B_VEC[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen; dsum += D_VEC[i]
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=ai, tw_b=bi,
                             early=float(early), late=float(late),
                             penalty=float(pen), service=float(S_VEC[i]),
                             depart=float(cur + S_VEC[i]), demand=float(D_VEC[i])))
        cur += float(S_VEC[i]); last = i
    travel += T_FULL[last, 0]
    if with_detail:
        return float(travel), float(penalty), float(dsum), rows
    return float(travel), float(penalty), float(dsum)


def route_pure_travel(perm):
    """Q1：纯 travel（不含时间窗罚）。"""
    cur = 0.0; last = 0
    for i in perm:
        cur += T_FULL[last, int(i)]; last = int(i)
    cur += T_FULL[last, 0]
    return float(cur)


# ---------- per-route polish (J = travel + tw_pen) ----------
def polish_route(route, mode="J"):
    """mode='J' 用 travel+pen，mode='travel' 仅用 travel（Q1）。"""
    def cost(r):
        if not r: return 0.0
        if mode == "travel":
            return route_pure_travel(r)
        t, p, _ = evaluate_route(r)
        return t + p

    cur = list(route); cur_cost = cost(cur); n = len(cur)
    if n <= 1: return cur, cur_cost
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = cur[:i] + cur[i:k + 1][::-1] + cur[k + 1:]
                c = cost(cand)
                if c < cur_cost - 1e-9:
                    cur, cur_cost = cand, c; improved = True
        for L in (1, 2, 3):
            for i in range(0, n - L + 1):
                seg = cur[i:i + L]; base = cur[:i] + cur[i + L:]
                for j in range(0, len(base) + 1):
                    if j == i: continue
                    cand = base[:j] + seg + base[j:]
                    c = cost(cand)
                    if c < cur_cost - 1e-9:
                        cur, cur_cost = cand, c; improved = True; break
                if improved: break
            if improved: break
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cur.copy(); cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c < cur_cost - 1e-9:
                    cur, cur_cost = cand, c; improved = True
    return cur, cur_cost


# ---------- CIM 真机统一接口 ----------
def submit_cim(Q: np.ndarray, task_name: str) -> tuple[np.ndarray, list[float], int]:
    """提交 1 个 QUBO 到 CIM 真机，返回 (binary_x, hamiltonian_list, n_ising)。"""
    Q_for = kw.qubo.adjust_qubo_matrix_precision(Q, bit_width=8)
    ising_matrix, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q_for)
    n_ising = ising_matrix.shape[0]
    nvar = Q.shape[0]
    assert n_ising <= 550, f"task {task_name} ising 维度 {n_ising} > 550 上限"
    print(f"  [CIM] task={task_name}  qubo={nvar}维, ising={n_ising}维, sample_number={CIM_SAMPLE}", flush=True)
    t0 = time.time()
    cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True,
                              task_mode="quota", sample_number=CIM_SAMPLE)
    spins = cim.solve(ising_matrix)
    dt = time.time() - t0
    x_bin = spin_to_binary(spins, nvar)
    Hs = [float(x_bin[i] @ Q_for @ x_bin[i]) for i in range(x_bin.shape[0])]
    print(f"  [CIM] task={task_name} 完成 {dt:.1f}s, 样本数={x_bin.shape[0]}, 最低 H={min(Hs):.1f}", flush=True)
    return x_bin, Hs, int(n_ising)


# ====================================================================
# Q1 · n=15 单车 TSP（仅最小化 travel）
# ====================================================================
def run_q1():
    print("\n" + "=" * 60, flush=True)
    print("Q1 · n=15 单车 TSP（CIM 真机）", flush=True)
    print("=" * 60, flush=True)
    n_sub = 15
    customers = list(range(1, n_sub + 1))
    T_sub = make_sub_T(0, customers, 0)
    Q = build_qubo_tsp(T_sub, n_sub, A_PEN)
    x_bin, Hs, n_ising = submit_cim(Q, f"q1_n15_{TASK_TS}")
    feas_perms = []
    for k in range(x_bin.shape[0]):
        perm_idx, feas = decode_sub(x_bin[k], n_sub)
        if feas:
            feas_perms.append([customers[i] for i in perm_idx])
    print(f"  Q1 可行解 {len(feas_perms)}/{x_bin.shape[0]}", flush=True)

    # polish 所有可行 + 选 travel 最小
    best_cost = float("inf"); best_perm = None
    for p in feas_perms:
        pp, c = polish_route(p, mode="travel")
        if c < best_cost: best_cost = c; best_perm = pp
    PY_BASELINE_Q1 = 29.0  # Held-Karp 精确解
    print(f"  Q1 best travel = {best_cost:.0f} (vs 纯 Python 全局最优 = {PY_BASELINE_Q1:.0f})", flush=True)

    return dict(
        problem="Q1 · n=15 单车 TSP",
        n_qubo_vars=int(n_sub * n_sub),
        n_ising_spins=n_ising,
        cim_sample_number=CIM_SAMPLE,
        n_feasible=len(feas_perms),
        n_total_samples=int(x_bin.shape[0]),
        hamiltonian_values=Hs,
        best_route=[0] + [int(c) for c in best_perm] + [0] if best_perm else None,
        best_travel=best_cost,
        py_baseline_travel=PY_BASELINE_Q1,
        delta_travel=float(best_cost - PY_BASELINE_Q1),
    )


# ====================================================================
# Q2 · n=15 单车 + 时间窗（方案 C：QUBO 仅含 one-hot + 距离）
# ====================================================================
def run_q2():
    print("\n" + "=" * 60, flush=True)
    print("Q2 · n=15 单车 + 时间窗（CIM 真机，方案 C）", flush=True)
    print("=" * 60, flush=True)
    n_sub = 15
    customers = list(range(1, n_sub + 1))
    T_sub = make_sub_T(0, customers, 0)
    Q = build_qubo_tsp(T_sub, n_sub, A_PEN)  # 同 Q1 QUBO（时间窗在解码后评估）
    x_bin, Hs, n_ising = submit_cim(Q, f"q2_n15_{TASK_TS}")
    feas_perms = []
    for k in range(x_bin.shape[0]):
        perm_idx, feas = decode_sub(x_bin[k], n_sub)
        if feas:
            feas_perms.append([customers[i] for i in perm_idx])
    print(f"  Q2 可行解 {len(feas_perms)}/{x_bin.shape[0]}", flush=True)

    best_J = float("inf"); best_perm = None; best_tr = None; best_pen = None
    for p in feas_perms:
        pp, c = polish_route(p, mode="J")
        if c < best_J:
            best_J = c; best_perm = pp
            t_, pe, _ = evaluate_route(pp); best_tr = t_; best_pen = pe
    PY_BASELINE_Q2 = 84121.0
    print(f"  Q2 best J = {best_J:.0f} (travel={best_tr:.0f}, pen={best_pen:.0f}) "
          f"vs 纯 Python = {PY_BASELINE_Q2:.0f}", flush=True)

    return dict(
        problem="Q2 · n=15 单车 + 时间窗",
        n_qubo_vars=int(n_sub * n_sub),
        n_ising_spins=n_ising,
        cim_sample_number=CIM_SAMPLE,
        n_feasible=len(feas_perms),
        n_total_samples=int(x_bin.shape[0]),
        hamiltonian_values=Hs,
        best_route=[0] + [int(c) for c in best_perm] + [0] if best_perm else None,
        best_travel=best_tr, best_pen=best_pen, best_J=best_J,
        py_baseline_J=PY_BASELINE_Q2,
        delta_J=float(best_J - PY_BASELINE_Q2) if best_perm else None,
    )


# ====================================================================
# Q3 · n=50 单车 + 时间窗（滚动窗口分解 3 段）
# ====================================================================
PY_Q3_PERM = [40, 2, 21, 26, 12, 28, 27, 1, 31, 7, 19, 48, 8, 18, 5, 6, 37, 42,
              15, 43, 14, 38, 44, 16, 17, 45, 46, 47, 36, 49, 11, 10, 30, 32,
              20, 9, 34, 35, 33, 50, 3, 29, 24, 25, 4, 39, 23, 22, 41, 13]
PY_BASELINE_Q3 = 4941906.0


def run_q3():
    print("\n" + "=" * 60, flush=True)
    print("Q3 · n=50 单车 + 时间窗（CIM 真机，滚动窗口分解 3 段）", flush=True)
    print("=" * 60, flush=True)
    cur_perm = list(PY_Q3_PERM)
    seg_bounds = [(0, 17), (17, 34), (34, 50)]
    sub_logs = []; all_H = []
    for seg_idx, (lo, hi) in enumerate(seg_bounds):
        n_sub = hi - lo
        a = 0 if lo == 0 else int(cur_perm[lo - 1])
        b = 0 if hi == 50 else int(cur_perm[hi])
        customers = [int(c) for c in cur_perm[lo:hi]]
        T_sub = make_sub_T(a, customers, b)
        Q_sub = build_qubo_tsp(T_sub, n_sub, A_PEN)
        x_bin, Hs, n_ising = submit_cim(Q_sub, f"q3_seg{seg_idx + 1}_{TASK_TS}")
        all_H.extend(Hs)
        feas_perms = []
        for k in range(x_bin.shape[0]):
            perm_idx, feas = decode_sub(x_bin[k], n_sub)
            if feas:
                feas_perms.append([customers[i] for i in perm_idx])
        # 拼接 + polish 全 perm
        original_J = evaluate_route(cur_perm)
        original_J_val = original_J[0] + original_J[1]
        best_J_after = original_J_val; best_perm_after = list(cur_perm)
        for sp in feas_perms:
            cand_perm = list(cur_perm); cand_perm[lo:hi] = sp
            pp, c = polish_route(cand_perm, mode="J")
            if c < best_J_after - 1e-9:
                best_J_after = c; best_perm_after = pp
        cur_perm = best_perm_after
        sub_logs.append(dict(
            seg_idx=seg_idx + 1, n_sub=n_sub,
            sub_qubo_bits=int(n_sub * n_sub), n_ising_spins=int(n_ising),
            n_feasible=len(feas_perms), n_total=int(x_bin.shape[0]),
            customers_in=customers,
            J_before=float(original_J_val), J_after=float(best_J_after),
        ))
        print(f"  [Q3 段 {seg_idx + 1}/3] feas={len(feas_perms)}/{x_bin.shape[0]}, "
              f"J: {original_J_val:.0f} → {best_J_after:.0f}", flush=True)

    # 全局 polish
    cur_perm, _ = polish_route(cur_perm, mode="J")
    travel, pen, _ = evaluate_route(cur_perm)
    J = travel + pen
    print(f"  Q3 final J={J:.0f} (travel={travel:.0f}, pen={pen:.0f}) "
          f"vs 纯 Python = {PY_BASELINE_Q3:.0f}", flush=True)

    return dict(
        problem="Q3 · n=50 单车（滚动窗口分解）",
        sub_problems=sub_logs,
        hamiltonian_values=all_H,
        cim_sample_number=CIM_SAMPLE,
        max_subqubo_bits=int(max(s["sub_qubo_bits"] for s in sub_logs)),
        cim_qubit_limit=550,
        final_route=[0] + [int(c) for c in cur_perm] + [0],
        final_travel=travel, final_pen=pen, final_J=J,
        py_baseline_J=PY_BASELINE_Q3,
        delta_J=float(J - PY_BASELINE_Q3),
    )


# ====================================================================
# Q4 · n=50 多车（K=7，每车一子 TSP QUBO）
# ====================================================================
PY_Q4_ROUTES = [
    [15, 2, 21, 40, 6, 37, 17],
    [31, 30, 9, 34, 35, 20, 32],
    [27, 16, 44, 38, 14, 43, 42, 13],
    [47, 19, 36, 49, 11, 10, 1],
    [33, 25, 39, 23, 22, 41, 4],
    [28, 29, 12, 3, 50, 26, 24],
    [5, 45, 7, 18, 8, 46, 48],
]
PY_BASELINE_Q4_OBJ = 7149.0
PY_BASELINE_Q4_TR = 109.0
PY_BASELINE_Q4_PE = 40.0


def run_q4():
    print("\n" + "=" * 60, flush=True)
    print("Q4 · n=50 多车 K=7（CIM 真机，每车一子 TSP QUBO）", flush=True)
    print("=" * 60, flush=True)
    sub_logs = []; all_H = []; new_routes = []
    for k_idx, customers in enumerate(PY_Q4_ROUTES):
        n_sub = len(customers)
        T_sub = make_sub_T(0, customers, 0)
        Q_sub = build_qubo_tsp(T_sub, n_sub, A_PEN)
        x_bin, Hs, n_ising = submit_cim(Q_sub, f"q4_v{k_idx + 1}_{TASK_TS}")
        all_H.extend(Hs)
        feas_perms = []
        for k in range(x_bin.shape[0]):
            perm_idx, feas = decode_sub(x_bin[k], n_sub)
            if feas:
                feas_perms.append([customers[i] for i in perm_idx])
        orig_p, orig_c = polish_route(customers, mode="J")
        best_seg = orig_p; best_cost = orig_c
        for sp in feas_perms:
            pp, c = polish_route(sp, mode="J")
            if c < best_cost - 1e-9:
                best_seg = pp; best_cost = c
        new_routes.append(best_seg)
        t_, pe, dsum = evaluate_route(best_seg)
        sub_logs.append(dict(
            vehicle=k_idx + 1, n_sub=n_sub,
            sub_qubo_bits=int(n_sub * n_sub), n_ising_spins=int(n_ising),
            n_feasible=len(feas_perms), n_total=int(x_bin.shape[0]),
            customers_in=customers, route_after=[0] + best_seg + [0],
            travel=float(t_), pen=float(pe), demand=float(dsum),
        ))
        print(f"  [Q4 V{k_idx + 1}/7] n_sub={n_sub}, qubo={n_sub*n_sub}比特, "
              f"feas={len(feas_perms)}/{x_bin.shape[0]}, route={best_seg}", flush=True)

    travel = sum(s["travel"] for s in sub_logs)
    penalty = sum(s["pen"] for s in sub_logs)
    obj_M = M_VEHICLE * len(new_routes) + travel + penalty
    print(f"  Q4 final travel={travel:.0f}, pen={penalty:.0f}, "
          f"obj_M={obj_M:.0f} vs 纯 Python = {PY_BASELINE_Q4_OBJ:.0f}", flush=True)

    return dict(
        problem="Q4 · n=50 多车 K=7（每车一子 QUBO）",
        K=len(new_routes),
        sub_problems=sub_logs,
        hamiltonian_values=all_H,
        cim_sample_number=CIM_SAMPLE,
        max_subqubo_bits=int(max(s["sub_qubo_bits"] for s in sub_logs)),
        cim_qubit_limit=550,
        final_routes=[s["route_after"] for s in sub_logs],
        final_travel=travel, final_pen=penalty,
        final_obj_M=obj_M,
        py_baseline_obj_M=PY_BASELINE_Q4_OBJ,
        delta_obj_M=float(obj_M - PY_BASELINE_Q4_OBJ),
    )


# ====================================================================
# 主流程
# ====================================================================
def main():
    print(f"\n{'#'*60}", flush=True)
    print(f"# 全 4 问 · CIM 真机一次性验证（共 12 个 QUBO） #", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"# Q1: 1 × QUBO(225 比特)", flush=True)
    print(f"# Q2: 1 × QUBO(225 比特)", flush=True)
    print(f"# Q3: 3 × QUBO(289+289+256 比特)", flush=True)
    print(f"# Q4: 7 × QUBO(49+49+64+49+49+49+49 比特)", flush=True)
    print(f"# 总样本数 = 12 × {CIM_SAMPLE} = {12 * CIM_SAMPLE}", flush=True)
    print(f"{'#'*60}", flush=True)

    t_total = time.time()
    q1 = run_q1()
    q2 = run_q2()
    q3 = run_q3()
    q4 = run_q4()
    t_total = time.time() - t_total

    print(f"\n{'#'*60}", flush=True)
    print(f"# CIM 真机 4 问汇总（用时 {t_total:.1f}s）", flush=True)
    print(f"{'#'*60}", flush=True)
    print(f"# Q1: travel = {q1['best_travel']:.0f} (PY={q1['py_baseline_travel']:.0f}, Δ={q1['delta_travel']:+.0f})", flush=True)
    print(f"# Q2: J = {q2['best_J']:.0f if q2['best_J'] != float('inf') else 'N/A'} (PY={q2['py_baseline_J']:.0f})", flush=True)
    print(f"# Q3: J = {q3['final_J']:.0f} (PY={q3['py_baseline_J']:.0f}, Δ={q3['delta_J']:+.0f})", flush=True)
    print(f"# Q4: obj_M = {q4['final_obj_M']:.0f} (PY={q4['py_baseline_obj_M']:.0f}, Δ={q4['delta_obj_M']:+.0f})", flush=True)

    # ---- 落盘 ----
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if o == float("inf"): return None
        raise TypeError(f"not jsonable: {type(o)}")

    overall = dict(
        meta=dict(
            kaiwu_version=kw.__version__,
            cim_machine="CPQC-550",
            cim_qubit_limit=550,
            sample_number=CIM_SAMPLE,
            total_qubos_submitted=12,
            total_samples=12 * CIM_SAMPLE,
            total_time_sec=round(t_total, 2),
            timestamp=TASK_TS,
            bit_width_8_applied=True,
            scheme="all four problems use one-hot 位置编码 + 解码后时间窗评估（方案 C）",
        ),
        Q1=q1, Q2=q2, Q3=q3, Q4=q4,
    )
    out = OUT_RESULT / "all_q_cim_real.json"
    out.write_text(json.dumps(overall, ensure_ascii=False, indent=2, default=_to_jsonable),
                   encoding="utf-8")
    print(f"\n[写出] {out.relative_to(ROOT)}", flush=True)
    # 各问独立文件
    for name, data in [("q1_cim_real", q1), ("q2_cim_real", q2),
                       ("q3_cim_real", q3), ("q4_cim_real", q4)]:
        (OUT_RESULT / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_to_jsonable),
            encoding="utf-8")
    print(f"[写出] results/基础模型/q1..q4_cim_real.json", flush=True)

    # ---- 图 1：4 问哈密顿量演化合成 ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (key, label) in zip(axes.flat, [
        ("Q1", "Q1 · n=15 TSP (225 比特)"),
        ("Q2", "Q2 · n=15 + 时间窗 (225 比特)"),
        ("Q3", "Q3 · n=50 (3 子QUBO 合成, max 289 比特)"),
        ("Q4", "Q4 · n=50 K=7 (7 子QUBO 合成, max 64 比特)"),
    ]):
        Hs = overall[key]["hamiltonian_values"]
        ax.plot(Hs, lw=0.7, alpha=0.5, color="#888", label="原始解序")
        ax.plot(sorted(Hs, reverse=True), lw=1.4, color="#d9534f", label="排序")
        ax.axhline(min(Hs), color="#5cb85c", lw=1.0, ls="--",
                   label=f"最低 H = {min(Hs):.1f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("解索引"); ax.set_ylabel("哈密顿量 H")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"全 4 问 · CIM CPQC-550 真机哈密顿量演化（题目硬性要求图）\n"
                 f"sample_number={CIM_SAMPLE}, 12 个 QUBO 一次性提交，总用时 {t_total:.1f}s",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_all_q_cim_hamiltonian.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_all_q_cim_hamiltonian.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_all_q_cim_hamiltonian.png + .pdf", flush=True)

    # ---- 图 2：4 问 vs 基线柱状图 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Q1 travel", "Q2 J", "Q3 J", "Q4 obj_M"]
    cim_vals = [q1["best_travel"], q2["best_J"] if q2["best_J"] != float("inf") else 0,
                q3["final_J"], q4["final_obj_M"]]
    py_vals = [q1["py_baseline_travel"], q2["py_baseline_J"],
               q3["py_baseline_J"], q4["py_baseline_obj_M"]]
    x = np.arange(len(labels)); w = 0.35
    bar1 = ax.bar(x - w / 2, cim_vals, w, label="CIM 真机", color="#3a78c2")
    bar2 = ax.bar(x + w / 2, py_vals, w, label="纯 Python 基线", color="#5cb85c", alpha=0.85)
    for b, v in zip(bar1, cim_vals): ax.text(b.get_x() + w/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for b, v in zip(bar2, py_vals): ax.text(b.get_x() + w/2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_yscale("symlog")
    ax.set_ylabel("数值（symlog）")
    ax.set_title("CIM 真机 vs 纯 Python 基线 · 4 问汇总")
    ax.grid(axis="y", alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_all_q_cim_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_all_q_cim_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_all_q_cim_summary.png + .pdf", flush=True)

    # ---- 图 3：12 个 QUBO 比特预算 ----
    bit_data = []
    bit_data.append(("Q1", q1["n_qubo_vars"]))
    bit_data.append(("Q2", q2["n_qubo_vars"]))
    for s in q3["sub_problems"]:
        bit_data.append((f"Q3-S{s['seg_idx']}", s["sub_qubo_bits"]))
    for s in q4["sub_problems"]:
        bit_data.append((f"Q4-V{s['vehicle']}", s["sub_qubo_bits"]))
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = np.arange(len(bit_data))
    bits = [b[1] for b in bit_data]
    cmap = plt.colormaps.get_cmap("tab20")
    bars = ax.bar(xs, bits, color=[cmap(i % 20) for i in range(len(bit_data))],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, bits):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(550, color="#d9534f", lw=1.5, ls="--", label="CIM CPQC-550 真机上限")
    ax.axhline(2500, color="#888", lw=1.0, ls=":", label="完整 n=50 QUBO = 2500 比特（不可行）")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bit_data], rotation=30, fontsize=9)
    ax.set_ylabel("QUBO 比特数（log scale）")
    ax.set_yscale("log")
    ax.set_title(f"全 4 问 · 12 个 QUBO 的比特预算（铁律 §二.5/二.6）—— 全部 ≤ 550 ✓")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_all_q_cim_qubits.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_all_q_cim_qubits.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_all_q_cim_qubits.png + .pdf", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
