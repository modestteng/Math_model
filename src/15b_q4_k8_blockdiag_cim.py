"""
问题四 K=8 副方案 · 一次性 CIM 真机验证（block-diagonal 合并）

策略：
  - v1 已单独跑过（cim_q4_k8_v1_20260426_210949.json），结果复用
  - v2..v8 拼成 block-diagonal QUBO：49+49+36+64+25+49+25 = 297 维
  - 8-bit 精度 + ising 转换：298 ising 自旋（≤550 真机上限 ✓）
  - 1 次 cim.solve → 解码 7 段 → 各段 polish → 与 v1 合并 → 总 obj_M
  - 配额消耗：1 次（v1 已扣 1 次，本次 1 次，K=8 全方案累计 2 次配额）

对比 R-Q4-005 SDK SA：每车独立 SA，7 次 cim.solve
本脚本：1 次 cim.solve 完成 7 个子问题（block-diag 优势）

输出
  results/真机结果/q4/cim_q4_k8_blockdiag_<TS>.json   单文件含全部解码 + 总对比
"""
from __future__ import annotations
import os
for _k in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

import json, time, tempfile, glob
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
M_VEHICLE = 1000.0
CIM_SAMPLE = 10
TS = time.strftime("%Y%m%d_%H%M%S")

# K=8 方案 8 辆车的客户分配
K8_ROUTES = [
    [27, 2, 40, 3, 50],            # v1 (5 客户, 25 比特) — 已跑
    [31, 30, 35, 34, 9, 20, 32],   # v2 (7, 49)
    [15, 42, 16, 44, 37, 6, 48],   # v3 (7, 49)
    [11, 19, 47, 49, 10, 1],       # v4 (6, 36)
    [28, 21, 41, 22, 23, 39, 25, 4],  # v5 (8, 64)
    [33, 29, 12, 24, 26],          # v6 (5, 25)
    [36, 7, 18, 8, 46, 45, 17],    # v7 (7, 49)
    [5, 38, 14, 43, 13],           # v8 (5, 25)
]
PY_K8_TRAVEL = 114.0
PY_K8_PEN = 10.0
PY_K8_J_INNER = 124.0
PY_K8_OBJ_M = 8124.0

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
print(f"[Kaiwu] license OK", flush=True)
CKPT_DIR = Path(tempfile.gettempdir()) / "kaiwu_cim_ckpt_q4_k8_blockdiag"
CKPT_DIR.mkdir(exist_ok=True)
kw.common.CheckpointManager.save_dir = str(CKPT_DIR)


# === Step 1：读取 v1 结果（已跑） ===
v1_files = sorted(glob.glob(str(OUT_REAL / "q4" / "cim_q4_k8_v1_*.json")))
v1_files = [f for f in v1_files if "ERROR" not in f]
assert v1_files, "找不到 v1 真机结果"
v1_result = json.loads(Path(v1_files[-1]).read_text(encoding="utf-8"))
print(f"\n[v1 复用] {Path(v1_files[-1]).name}", flush=True)
print(f"  cost={v1_result['seg_cost']}, travel={v1_result['seg_travel']}, pen={v1_result['seg_penalty']}", flush=True)

# === Step 2：构 v2..v8 子 QUBO，block-diag 合并 ===
print(f"\n[block-diag 构造]", flush=True)
sub_qubos = []
sub_metas = []
total_nvar = 0
for v_idx in range(2, 9):
    customers = K8_ROUTES[v_idx - 1]
    n_sub = len(customers)
    T_sub = make_sub_T(customers)
    Q = build_subqubo_tsp(T_sub, n_sub, A_PEN)
    sub_qubos.append(Q)
    sub_metas.append(dict(v_idx=v_idx, customers=customers, n_sub=n_sub,
                         nvar=n_sub * n_sub, offset=total_nvar))
    print(f"  v{v_idx}: n_sub={n_sub}, nvar={n_sub*n_sub}, offset={total_nvar}", flush=True)
    total_nvar += n_sub * n_sub

print(f"  合并 QUBO 维度 = {total_nvar}", flush=True)

# block-diagonal 合并
Q_big = np.zeros((total_nvar, total_nvar))
for meta, Q_sub in zip(sub_metas, sub_qubos):
    o = meta["offset"]; n = meta["nvar"]
    Q_big[o:o + n, o:o + n] = Q_sub

# 8-bit 精度
print(f"\n[8-bit 精度调整]", flush=True)
Q_big_8bit = kw.qubo.adjust_qubo_matrix_precision(Q_big, bit_width=8)
print(f"  8-bit 范围 [{int(Q_big_8bit.min())}, {int(Q_big_8bit.max())}]", flush=True)

# ising 转换
print(f"\n[ising 转换]", flush=True)
ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_big_8bit)
n_ising = ising_mat.shape[0]
print(f"  ising 维度 = {n_ising} (含 1 辅助自旋)", flush=True)
print(f"  bias = {ising_bias:.1f}", flush=True)
assert n_ising <= 550, f"ising {n_ising} > 550 真机上限"

# === Step 3：1 次 cim.solve ===
task_name = f"q4_k8_blockdiag_{TS}"
print(f"\n[CIM] 提交 1 次 task={task_name}, sample_number={CIM_SAMPLE}", flush=True)
print(f"  消耗配额 = 1 次（注意：本次只调 1 次 cim.solve 解 7 个子问题）", flush=True)
t0 = time.time()
cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True,
                          task_mode="quota", sample_number=CIM_SAMPLE)
spins = cim.solve(ising_mat)
dt = time.time() - t0
print(f"  [CIM] 完成 {dt:.1f}s, spins.shape={spins.shape}", flush=True)

# === Step 4：解码 7 段 ===
x_bin = spin_to_binary(spins, total_nvar)  # (n_samples, total_nvar)
print(f"  解码：x_bin.shape={x_bin.shape}", flush=True)

# 整体哈密顿量
Hs_total = []
for k in range(spins.shape[0]):
    s_k = spins[k] if spins.shape[1] == total_nvar + 1 else np.append(spins[k], 1)
    h_k = float(s_k @ ising_mat @ s_k)
    Hs_total.append(h_k)
print(f"  整体 H ∈ [{min(Hs_total):.0f}, {max(Hs_total):.0f}]", flush=True)

# 逐段解码 + polish
seg_results = []
print(f"\n[逐段解码 + polish]", flush=True)
for meta in sub_metas:
    v_idx = meta["v_idx"]; customers = meta["customers"]; n_sub = meta["n_sub"]
    o = meta["offset"]; nvar = meta["nvar"]
    print(f"  v{v_idx}: 解码段 [{o}:{o+nvar}]", flush=True)

    feas_perms = []
    for k in range(x_bin.shape[0]):
        x_seg = x_bin[k, o:o + nvar]
        perm_idx, feas = decode_sub(x_seg, n_sub)
        if feas:
            feas_perms.append(perm_idx)

    # polish 取最优
    best_seg = None; best_cost = None
    for sp in feas_perms:
        seg = [customers[i] for i in sp]
        seg_p, c = polish_route(seg)
        if best_cost is None or c < best_cost - 1e-9:
            best_seg = seg_p; best_cost = c
    if best_seg is None:
        best_seg, best_cost = polish_route(customers)
    tr, pn = evaluate_route(best_seg)

    seg_result = dict(
        v_idx=v_idx,
        n_sub=n_sub, nvar_block=nvar, offset=o,
        n_feasible=len(feas_perms),
        feasibility_rate=round(len(feas_perms) / x_bin.shape[0], 4),
        customers_in=list(customers),
        best_seg_after_polish=[int(c) for c in best_seg],
        best_route_with_depot=[0] + [int(c) for c in best_seg] + [0],
        seg_travel=float(tr), seg_penalty=float(pn),
        seg_cost=float(tr + pn),
        seg_demand=float(sum(D_VEC[c] for c in best_seg)),
    )
    seg_results.append(seg_result)
    print(f"    feas={len(feas_perms)}/{x_bin.shape[0]}, cost={tr+pn:.0f} (travel={tr:.0f}+pen={pn:.0f})", flush=True)

# === Step 5：合并 v1 + v2..v8，算总 obj_M ===
all_segs = [dict(v_idx=1,
                 n_sub=v1_result["n_qubo_vars"],
                 nvar_block=v1_result["n_qubo_vars"],
                 offset=-1,  # 单独提交
                 n_feasible=v1_result["n_feasible"],
                 feasibility_rate=v1_result["feasibility_rate"],
                 customers_in=v1_result["customers_in"],
                 best_seg_after_polish=v1_result["best_seg_after_polish"],
                 best_route_with_depot=v1_result["best_route_with_depot"],
                 seg_travel=v1_result["seg_travel"],
                 seg_penalty=v1_result["seg_penalty"],
                 seg_cost=v1_result["seg_cost"],
                 seg_demand=v1_result["seg_demand"])] + seg_results

total_travel = sum(s["seg_travel"] for s in all_segs)
total_pen = sum(s["seg_penalty"] for s in all_segs)
total_J_inner = total_travel + total_pen
total_obj_M = 8 * M_VEHICLE + total_J_inner

print(f"\n{'='*60}", flush=True)
print(f"K=8 真机求解汇总（block-diag 一次提交 + v1 复用）：", flush=True)
print(f"{'='*60}", flush=True)
print(f"  CIM 真机 K=8: travel={total_travel:.0f}, pen={total_pen:.0f}, J_inner={total_J_inner:.0f}, obj_M={total_obj_M:.0f}", flush=True)
print(f"  Python K=8  : travel={PY_K8_TRAVEL:.0f}, pen={PY_K8_PEN:.0f}, J_inner={PY_K8_J_INNER:.0f}, obj_M={PY_K8_OBJ_M:.0f}", flush=True)
gap = (total_obj_M - PY_K8_OBJ_M) / PY_K8_OBJ_M * 100 if PY_K8_OBJ_M > 0 else 0
print(f"  gap = {total_obj_M - PY_K8_OBJ_M:+.0f} ({gap:+.2f}%)", flush=True)
print(f"  配额消耗：1 次（v1 已扣 1 次，K=8 全方案累计 2 次）", flush=True)

# === Step 6：落盘 ===
result = dict(
    timestamp=TS,
    user_id=KAIWU_USER_ID,
    scheme="K=8_blockdiag",
    cim_machine="CPQC-550",
    cim_task_name=task_name,
    cim_solve_time_sec=round(dt, 2),
    cim_sample_number=CIM_SAMPLE,
    cim_quota_consumed=1,
    n_qubo_vars_combined=int(total_nvar),
    n_ising_spins=int(n_ising),
    bit_width_8_applied=True,
    n_total_samples=int(x_bin.shape[0]),
    hamiltonian_total=Hs_total,
    min_hamiltonian_total=float(min(Hs_total)),
    max_hamiltonian_total=float(max(Hs_total)),
    py_baseline=dict(travel=PY_K8_TRAVEL, penalty=PY_K8_PEN,
                     J_inner=PY_K8_J_INNER, obj_M=PY_K8_OBJ_M),
    cim_summary=dict(travel=float(total_travel), penalty=float(total_pen),
                     J_inner=float(total_J_inner), obj_M=float(total_obj_M),
                     gap_to_py=float(total_obj_M - PY_K8_OBJ_M)),
    segments=all_segs,
)
out_path = OUT_REAL / "q4" / f"cim_q4_k8_blockdiag_{TS}.json"


def _to_jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not jsonable: {type(o)}")


out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
                    encoding="utf-8")
print(f"\n[写出] {out_path.relative_to(ROOT)}", flush=True)
print(f"\n[ALL DONE]", flush=True)
