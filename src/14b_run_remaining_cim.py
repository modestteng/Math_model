"""
对还没真机验证的 9 个矩阵跑 CIM（SDK 接口）。
跳过 Q1, Q2（已有真机结果），跑 Q3-seg1/seg2/seg3 + Q4-v1..v7。

读源：results/基础模型/qubo_matrices/csv_ising_8bit/<name>_ising.csv
写源：results/真机结果/q{3,4}/cim_<name>_<timestamp>.json   每跑完一个立即落盘

为避免之前 kill 时丢数据：
  ✓ 每个 task 完成后立即写 JSON
  ✓ 主循环捕获异常，不让单个 task 崩溃中断后续
  ✓ Last-task progress 实时打印
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
SRC_NPY = ROOT / "results/基础模型/qubo_matrices"
SRC_CSV = ROOT / "results/基础模型/qubo_matrices/csv_ising_8bit"
OUT_REAL = ROOT / "results/真机结果"

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"
A_PEN = 200.0
M1, M2 = 10.0, 20.0
CIM_SAMPLE = 10
TS = time.strftime("%Y%m%d_%H%M%S")

# 数据
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_FULL = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int).astype(float)
A_VEC = nodes_raw["tw_a"].values.astype(float)
B_VEC = nodes_raw["tw_b"].values.astype(float)
S_VEC = nodes_raw["service"].values.astype(float)
D_VEC = nodes_raw["demand"].values.astype(float)

# Q3 baseline perm（拼回用）
PY_Q3_PERM = [40, 2, 21, 26, 12, 28, 27, 1, 31, 7, 19, 48, 8, 18, 5, 6, 37, 42,
              15, 43, 14, 38, 44, 16, 17, 45, 46, 47, 36, 49, 11, 10, 30, 32,
              20, 9, 34, 35, 33, 50, 3, 29, 24, 25, 4, 39, 23, 22, 41, 13]


def evaluate_route(route, with_detail=False):
    travel = 0.0; pen = 0.0; cur = 0.0; last = 0; rows = []
    for i in route:
        i = int(i); tt = T_FULL[last, i]; cur += tt; travel += tt
        ai, bi = float(A_VEC[i]), float(B_VEC[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        p = M1 * early ** 2 + M2 * late ** 2
        pen += p
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=ai, tw_b=bi,
                             early=float(early), late=float(late),
                             penalty=float(p), service=float(S_VEC[i]),
                             depart=float(cur + S_VEC[i]),
                             demand=float(D_VEC[i])))
        cur += float(S_VEC[i]); last = i
    travel += T_FULL[last, 0]
    if with_detail:
        return float(travel), float(pen), rows
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
CKPT_DIR = Path(tempfile.gettempdir()) / "kaiwu_cim_ckpt_remaining"
CKPT_DIR.mkdir(exist_ok=True)
kw.common.CheckpointManager.save_dir = str(CKPT_DIR)
print(f"[Kaiwu] CheckpointManager.save_dir = {CKPT_DIR}", flush=True)


def submit_one(name: str, n_qubo: int, customers: list):
    """提交一个矩阵到 CIM，立即落盘结果。"""
    print(f"\n========== {name} (n_qubo={n_qubo}) ==========", flush=True)
    csv_path = SRC_CSV / f"{name}_ising.csv"
    ising_mat = np.loadtxt(csv_path, delimiter=",").astype(int).astype(float)
    n_ising = ising_mat.shape[0]
    print(f"  读 CSV {csv_path.name}：ising {n_ising}×{n_ising}", flush=True)
    assert n_ising <= 550, f"{name} ising {n_ising} > 550"

    task_name = f"{name}_{TS}"
    print(f"  [CIM] 提交 task={task_name}, sample_number={CIM_SAMPLE} ...", flush=True)
    t0 = time.time()
    cim = kw.cim.CIMOptimizer(task_name=task_name, wait=True,
                              task_mode="quota", sample_number=CIM_SAMPLE)
    spins = cim.solve(ising_mat)
    dt = time.time() - t0
    print(f"  [CIM] 完成 {dt:.1f}s, spins.shape={spins.shape}", flush=True)

    x_bin = spin_to_binary(spins, n_qubo)
    Hs = []
    feas_perms = []
    for k in range(x_bin.shape[0]):
        # 注意 ising_mat 是已 8-bit + 转 ising 后的矩阵；H 用 ising 自旋形式更准
        # 这里用 binary x 在原 QUBO Q 上算 H 更直观，但 Q 没在内存（只有 ising），用 ising 算
        s_k = spins[k] if spins.shape[1] == n_qubo + 1 else np.append(spins[k], 1)
        h_k = float(s_k @ ising_mat @ s_k)  # ising 哈密顿量
        Hs.append(h_k)
        n_sub = int(round(np.sqrt(n_qubo)))
        perm_idx, feas = decode_sub(x_bin[k], n_sub)
        if feas:
            feas_perms.append(perm_idx)

    # 解码 + polish（按问题不同处理）
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
    )

    # 决定输出目录 + 解码
    if name.startswith("q3"):
        out_dir = OUT_REAL / "q3"
        # 拼接进 PY_Q3_PERM 同位置 + polish 全 perm
        seg_idx = int(name.split("seg")[1])  # 1, 2, 3
        seg_bounds = {1: (0, 17), 2: (17, 34), 3: (34, 50)}[seg_idx]
        lo, hi = seg_bounds
        py_perm_with_seg_replaced = list(PY_Q3_PERM)
        best_J_after = None; best_perm_after = None
        for sp in feas_perms:
            new_seg = [customers[i] for i in sp]
            cand = list(PY_Q3_PERM); cand[lo:hi] = new_seg
            cand_p, cc = polish_route(cand)
            if best_J_after is None or cc < best_J_after - 1e-9:
                best_J_after = cc; best_perm_after = cand_p
        if best_perm_after is None:
            # 无可行：用原 perm
            best_perm_after, best_J_after = polish_route(PY_Q3_PERM)
        tr, pn = evaluate_route(best_perm_after)
        result["seg_idx"] = seg_idx
        result["seg_bounds"] = list(seg_bounds)
        result["customers_in"] = list(customers)
        result["full_perm_after"] = [int(c) for c in best_perm_after]
        result["full_route"] = [0] + result["full_perm_after"] + [0]
        result["full_travel"] = float(tr)
        result["full_penalty"] = float(pn)
        result["full_J"] = float(tr + pn)
    elif name.startswith("q4"):
        out_dir = OUT_REAL / "q4"
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
        result["customers_in"] = list(customers)
        result["best_seg_after_polish"] = [int(c) for c in best_seg]
        result["best_route_with_depot"] = [0] + result["best_seg_after_polish"] + [0]
        result["seg_travel"] = float(tr)
        result["seg_penalty"] = float(pn)
        result["seg_cost"] = float(tr + pn)
        result["seg_demand"] = float(sum(D_VEC[c] for c in best_seg))
    else:
        out_dir = OUT_REAL  # 不应该到这里

    out_dir.mkdir(parents=True, exist_ok=True)
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
    if name.startswith("q3"):
        print(f"  Q3 全 J = {result['full_J']:.0f}", flush=True)
    elif name.startswith("q4"):
        print(f"  Q4 段 cost = {result['seg_cost']:.0f}, demand={result['seg_demand']:.0f}", flush=True)
    return result


# ============================================================
# 9 个矩阵的元信息（来自 metadata.json）
# ============================================================
metadata = json.loads((SRC_NPY / "metadata.json").read_text(encoding="utf-8"))
TARGETS = ["q3_seg2", "q3_seg3",
           "q4_v1", "q4_v2", "q4_v3", "q4_v4", "q4_v5", "q4_v6", "q4_v7"]
# 注意：q3_seg1 已在前次运行中跑成功（123.7s 完成）但因 reshape bug 解码失败，
# 配额已扣 1 次；spin 矩阵丢失故无法恢复；不再重跑（避免再扣）。
# 本次跑剩 9 个（q3_seg2/seg3 + Q4 七车）。

print(f"\n本次将提交 {len(TARGETS)} 个 task （新账号，每个消耗 1 次配额）：", flush=True)
print(f"  {TARGETS}", flush=True)

results_summary = []
for name in TARGETS:
    m = next((mm for mm in metadata if mm["name"] == name), None)
    if m is None:
        print(f"[SKIP] {name} 在 metadata 中找不到", flush=True); continue
    try:
        r = submit_one(name, m["n_qubo_vars"], m["customers"])
        results_summary.append(r)
    except Exception as e:
        print(f"\n[ERROR] {name} 出错：{e}", flush=True)
        traceback.print_exc()
        # 出错 task 也记录一条占位
        err_path = (OUT_REAL / ("q3" if name.startswith("q3") else "q4")
                   / f"cim_{name}_{TS}_ERROR.json")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(dict(name=name, error=str(e), timestamp=TS),
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        continue

# 汇总
print(f"\n{'='*60}\n本次共完成 {len(results_summary)}/{len(TARGETS)} 个 task\n{'='*60}", flush=True)
summary_path = OUT_REAL / f"batch_remaining_{TS}.json"
summary_path.write_text(json.dumps(dict(
    timestamp=TS, user_id=KAIWU_USER_ID,
    n_targets=len(TARGETS), n_completed=len(results_summary),
    targets=TARGETS,
    results=[{k: v for k, v in r.items() if k != "hamiltonian_values"}
             for r in results_summary],
), ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {summary_path.relative_to(ROOT)}", flush=True)
print("\n[ALL DONE]", flush=True)
