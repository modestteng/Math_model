"""
问题 4 副方案 K=8 的 CIM 真机验证（8 辆车独立子 TSP QUBO）。

链路一致性（铁律 §二.6）：
  - 起点：results/基础模型/q4_attack_optimal.json 的 K=8 方案（J_inner=124, obj_M=8124）
  - 8 辆车，每车客户数 5/7/7/6/8/5/7/5
  - 每车独立子 TSP QUBO：n²=25/49/49/36/64/25/49/25 ≤ 64 比特 ≤ 550 真机上限 ✓
  - 8-bit 精度（adjust_qubo_matrix_precision）+ ising 转换
  - 8 次 CIM 配额（每车 1 次 task）

输出
  results/真机结果/q4/cim_q4_k8_v{1..8}_<TS>.json   每车单独落盘
  results/真机结果/batch_q4_k8_<TS>.json            汇总
"""
from __future__ import annotations
import os
for _k in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

import json, time, tempfile, traceback
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT_REAL = ROOT / "results/真机结果"

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"
A_PEN = 200.0
M1, M2 = 10.0, 20.0
CIM_SAMPLE = 10
TS = time.strftime("%Y%m%d_%H%M%S")

# K=8 方案 8 辆车的客户分配（来自 q4_attack_optimal.json）
K8_ROUTES = [
    [27, 2, 40, 3, 50],
    [31, 30, 35, 34, 9, 20, 32],
    [15, 42, 16, 44, 37, 6, 48],
    [11, 19, 47, 49, 10, 1],
    [28, 21, 41, 22, 23, 39, 25, 4],
    [33, 29, 12, 24, 26],
    [36, 7, 18, 8, 46, 45, 17],
    [5, 38, 14, 43, 13],
]
PY_K8_TRAVEL = 114.0
PY_K8_PEN = 10.0
PY_K8_J_INNER = 124.0
PY_K8_OBJ_M = 8124.0  # 8*1000 + 124

# 数据
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_FULL = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int).astype(float)
A_VEC = nodes_raw["tw_a"].values.astype(float)
B_VEC = nodes_raw["tw_b"].values.astype(float)
S_VEC = nodes_raw["service"].values.astype(float)
D_VEC = nodes_raw["demand"].values.astype(float)


def evaluate_route(route):
    travel = 0.0; pen = 0.0; cur = 0.0; last = 0
    for i in route:
        i = int(i); tt = T_FULL[last, i]; cur += tt; travel += tt
        ai, bi = float(A_VEC[i]), float(B_VEC[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen += M1 * early ** 2 + M2 * late ** 2
        cur += float(S_VEC[i]); last = i
    travel += T_FULL[last, 0]
    return float(travel), float(pen)


def polish_route(route):
    def cost(r):
        if not r: return 0.0
        t, p = evaluate_route(r); return t + p
    cur = list(route); cc = cost(cur); n = len(cur)
    if n <= 1: return cur, cc
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = cur[:i] + cur[i:k + 1][::-1] + cur[k + 1:]
                c = cost(cand)
                if c < cc - 1e-9:
                    cur, cc = cand, c; improved = True
        for L in (1, 2, 3):
            for i in range(0, n - L + 1):
                seg = cur[i:i + L]; base = cur[:i] + cur[i + L:]
                for j in range(0, len(base) + 1):
                    if j == i: continue
                    cand = base[:j] + seg + base[j:]
                    c = cost(cand)
                    if c < cc - 1e-9:
                        cur, cc = cand, c; improved = True; break
                if improved: break
            if improved: break
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cur.copy(); cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c < cc - 1e-9:
                    cur, cc = cand, c; improved = True
    return cur, cc


def make_sub_T(customers):
    k = len(customers)
    T_sub = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        ci = customers[i - 1]
        T_sub[0, i] = T_FULL[0, ci]
        T_sub[i, 0] = T_FULL[ci, 0]
        for j in range(1, k + 1):
            if i != j:
                T_sub[i, j] = T_FULL[ci, customers[j - 1]]
    return T_sub


def build_subqubo_tsp(T_sub, n_sub, A_pen=A_PEN):
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


def decode_sub(x, n_sub):
    M = np.asarray(x).reshape(n_sub, n_sub)
    feas = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm_idx = [int(np.argmax(M[:, p])) for p in range(n_sub)]
    return perm_idx, feas


def spin_to_binary(spin, nvar):
    if spin.shape[1] == nvar + 1:
        s_aux = spin[:, -1:]; s_main = (spin * s_aux)[:, :-1]
    else:
        s_main = spin
    return ((s_main + 1) // 2).astype(np.int8)


# ---- Kaiwu ----
import kaiwu as kw
print(f"[Kaiwu] version = {kw.__version__}", flush=True)
kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
print(f"[Kaiwu] license OK (user_id={KAIWU_USER_ID})", flush=True)
CKPT_DIR = Path(tempfile.gettempdir()) / "kaiwu_cim_ckpt_q4_k8"
CKPT_DIR.mkdir(exist_ok=True)
kw.common.CheckpointManager.save_dir = str(CKPT_DIR)


def submit_vehicle(v_idx: int, customers: list):
    name = f"q4_k8_v{v_idx}"
    n_sub = len(customers)
    n_qubo = n_sub * n_sub
    print(f"\n========== {name} (n_customers={n_sub}, n_qubo={n_qubo}) ==========", flush=True)

    # 构子 QUBO
    T_sub = make_sub_T(customers)
    Q = build_subqubo_tsp(T_sub, n_sub, A_PEN)
    Q_8bit = kw.qubo.adjust_qubo_matrix_precision(Q, bit_width=8)
    ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_8bit)
    n_ising = ising_mat.shape[0]
    assert n_ising <= 550, f"{name} ising {n_ising} > 550"
    print(f"  QUBO {n_qubo}×{n_qubo}, ising {n_ising}×{n_ising}, bias={ising_bias:.1f}", flush=True)

    # 提交 CIM
    task_name = f"{name}_{TS}"
    print(f"  [CIM] 提交 task={task_name}, sample_number={CIM_SAMPLE} ...", flush=True)
    t0 = time.time()
    cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True,
                              task_mode="quota", sample_number=CIM_SAMPLE)
    spins = cim.solve(ising_mat)
    dt = time.time() - t0
    print(f"  [CIM] 完成 {dt:.1f}s, spins.shape={spins.shape}", flush=True)

    # 解析所有样本
    x_bin = spin_to_binary(spins, n_qubo)
    Hs = []; feas_perms = []
    for k in range(x_bin.shape[0]):
        s_k = spins[k] if spins.shape[1] == n_qubo + 1 else np.append(spins[k], 1)
        h_k = float(s_k @ ising_mat @ s_k)
        Hs.append(h_k)
        perm_idx, feas = decode_sub(x_bin[k], n_sub)
        if feas:
            feas_perms.append(perm_idx)

    # 单车 polish + 评估
    best_seg = None; best_cost = None
    for sp in feas_perms:
        seg = [customers[i] for i in sp]
        seg_p, c = polish_route(seg)
        if best_cost is None or c < best_cost - 1e-9:
            best_seg = seg_p; best_cost = c
    if best_seg is None:
        best_seg, best_cost = polish_route(customers)
    tr, pn = evaluate_route(best_seg)

    result = dict(
        name=name,
        cim_task_name=task_name,
        cim_machine="CPQC-550",
        cim_solve_time_sec=round(dt, 2),
        cim_sample_number=CIM_SAMPLE,
        n_qubo_vars=int(n_qubo),
        n_ising_spins=int(n_ising),
        bit_width_8_applied=True,
        n_total_samples=int(x_bin.shape[0]),
        n_feasible=len(feas_perms),
        feasibility_rate=round(len(feas_perms) / x_bin.shape[0], 4),
        hamiltonian_values=Hs,
        min_hamiltonian=float(min(Hs)),
        max_hamiltonian=float(max(Hs)),
        timestamp=TS,
        vehicle_idx=v_idx,
        customers_in=list(customers),
        best_seg_after_polish=[int(c) for c in best_seg],
        best_route_with_depot=[0] + [int(c) for c in best_seg] + [0],
        seg_travel=float(tr),
        seg_penalty=float(pn),
        seg_cost=float(tr + pn),
        seg_demand=float(sum(D_VEC[c] for c in best_seg)),
    )

    out_dir = OUT_REAL / "q4"; out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"cim_{name}_{TS}.json"

    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"  [写出] {out_json.relative_to(ROOT)}", flush=True)
    print(f"  feas={len(feas_perms)}/{x_bin.shape[0]}, min H={result['min_hamiltonian']:.1f}", flush=True)
    print(f"  Q4 K=8 v{v_idx}: cost={result['seg_cost']:.0f} (travel={tr:.0f}+pen={pn:.0f}), demand={result['seg_demand']:.0f}", flush=True)
    return result


print(f"\n本次将为 K=8 方案 提交 8 个 task （每车 1 次配额）：", flush=True)
print(f"  Python 基线 K=8: travel={PY_K8_TRAVEL}, pen={PY_K8_PEN}, J_inner={PY_K8_J_INNER}, obj_M={PY_K8_OBJ_M}", flush=True)
for i, r in enumerate(K8_ROUTES, 1):
    print(f"  v{i}: {r} (n={len(r)}, demand={sum(D_VEC[c] for c in r):.0f})", flush=True)

results_summary = []
for v_idx, customers in enumerate(K8_ROUTES, 1):
    try:
        r = submit_vehicle(v_idx, customers)
        results_summary.append(r)
    except Exception as e:
        print(f"\n[ERROR] v{v_idx} 出错：{e}", flush=True)
        traceback.print_exc()
        err_path = OUT_REAL / "q4" / f"cim_q4_k8_v{v_idx}_{TS}_ERROR.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(dict(name=f"q4_k8_v{v_idx}", error=str(e), timestamp=TS),
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        continue

# 汇总
print(f"\n{'='*60}", flush=True)
print(f"K=8 真机求解：{len(results_summary)}/8 辆车完成", flush=True)
total_travel = sum(r["seg_travel"] for r in results_summary)
total_pen = sum(r["seg_penalty"] for r in results_summary)
total_cost = total_travel + total_pen
total_obj_M = 8 * 1000 + total_cost
print(f"  CIM 真机 K=8: travel={total_travel:.0f}, pen={total_pen:.0f}, J_inner={total_cost:.0f}, obj_M={total_obj_M:.0f}", flush=True)
print(f"  Python K=8  : travel={PY_K8_TRAVEL:.0f}, pen={PY_K8_PEN:.0f}, J_inner={PY_K8_J_INNER:.0f}, obj_M={PY_K8_OBJ_M:.0f}", flush=True)
gap = (total_obj_M - PY_K8_OBJ_M) / PY_K8_OBJ_M * 100 if PY_K8_OBJ_M > 0 else 0
print(f"  gap = {total_obj_M - PY_K8_OBJ_M:+.0f} ({gap:+.2f}%)", flush=True)

summary_path = OUT_REAL / f"batch_q4_k8_{TS}.json"
summary_path.write_text(json.dumps(dict(
    timestamp=TS, user_id=KAIWU_USER_ID, scheme="K=8",
    n_vehicles=8, n_completed=len(results_summary),
    py_baseline=dict(travel=PY_K8_TRAVEL, penalty=PY_K8_PEN,
                     J_inner=PY_K8_J_INNER, obj_M=PY_K8_OBJ_M),
    cim_summary=dict(travel=float(total_travel), penalty=float(total_pen),
                     J_inner=float(total_cost), obj_M=float(total_obj_M),
                     gap_to_py=float(total_obj_M - PY_K8_OBJ_M)),
    results=[{k: v for k, v in r.items() if k != "hamiltonian_values"}
             for r in results_summary],
), ensure_ascii=False, indent=2, default=lambda o: int(o) if isinstance(o, np.integer)
                                          else float(o) if isinstance(o, np.floating)
                                          else o.tolist() if isinstance(o, np.ndarray)
                                          else str(o)),
                       encoding="utf-8")
print(f"[写出] {summary_path.relative_to(ROOT)}", flush=True)
print("\n[ALL DONE]", flush=True)
