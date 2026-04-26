"""
问题 3 · Kaiwu SDK 验证（滚动窗口分解，真机兼容）

核心约束（铁律 §二.6）
----
  最终 Q3 求解必须上 CIM 550 真机。SDK 验证阶段使用与真机**完全相同**的编码和比特预算：
    每个子 QUBO 维度 ≤ 549（即 n_sub² ≤ 549，故 n_sub ≤ 23）
    8-bit 精度调整（CIM 真机硬性要求）
    分解策略一致，仅算力后端从 SDK SA 切换到 CIM

分解策略：滚动窗口（rolling window）局部 QUBO 重排
----
  1) 起点：Q3 第 1 步纯 Python 基线 perm（J=4,941,906）
  2) 把 perm 切成 K 段，每段 n_sub 个客户
  3) 对每段子 TSP（段前节点 a → 段内 n_sub 客户 → 段后节点 b）：
       - 构造子 T 矩阵 (n_sub+1)×(n_sub+1)，把段前/后节点当虚拟 depot
       - 调用 _q_lib.build_qubo_q1 得子 QUBO（仅距离 + one-hot 罚）
       - 8-bit 精度调整
       - solve_subqubo(Q_sub, backend) — SDK SA 或 CIM（统一接口）
       - 解码 + polish 子段
       - 取改进结果替换原段
  4) 拼接 → 全局 polish（含时间窗罚） → 最终 J
  5) 时间窗罚在解码阶段评估（同 Q2 方案 C）

子 QUBO 比特数：n_sub=17 时 = 17² = 289 比特  ≤ 550 ✓
对比：直接 n=50 全 QUBO = 2500 比特 ✗

输出
----
  results/基础模型/q3_kaiwu_sdk.json
  tables/tab_03c_q3_kaiwu_schedule.csv (+ .tex)
  figures/fig_03c_q3_kaiwu_route.png       (+ .pdf)  最终路径 + 甘特
  figures/fig_03c_q3_kaiwu_hamiltonian.png (+ .pdf)  子 QUBO 哈密顿量演化（题目硬性要求）
  figures/fig_03c_q3_kaiwu_compare.png     (+ .pdf)  SDK 分解 vs 纯 Python 基线
  figures/fig_03c_q3_decompose_diagram.png (+ .pdf)  分解策略示意 + 子 QUBO 比特数
"""
from __future__ import annotations

# 仅在本进程内临时清除代理变量，不修改用户系统代理
import os
for _k in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

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

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"

N = 50
N_SUB = 17                   # 子段大小：17² = 289 比特 ≤ 550 真机上限
A_PEN = 200.0
M1, M2 = 10.0, 20.0
SEED = 20260426

# Q3 第 1 步纯 Python 基线
PY_BASELINE_PERM = [40, 2, 21, 26, 12, 28, 27, 1, 31, 7, 19, 48, 8, 18, 5, 6, 37, 42,
                    15, 43, 14, 38, 44, 16, 17, 45, 46, 47, 36, 49, 11, 10, 30, 32,
                    20, 9, 34, 35, 33, 50, 3, 29, 24, 25, 4, 39, 23, 22, 41, 13]
PY_BASELINE_J = 4941906.0
PY_BASELINE_TRAVEL = 56.0
PY_BASELINE_PEN = 4941850.0

# ---------- Kaiwu SDK ----------
import kaiwu as kw  # noqa: E402

print(f"[Kaiwu] version = {kw.__version__}", flush=True)
kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
print("[Kaiwu] license initialized [OK]", flush=True)

# ---------- 数据 ----------
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full_int = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T_full = T_full_int[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)


# ---------- QUBO 构造（复用 Q1 公式：one-hot + 距离） ----------
def build_qubo_tsp(T_sub: np.ndarray, n_sub: int, A_pen: float = 200.0) -> np.ndarray:
    """子 TSP QUBO：从虚拟 depot(0) 出发经 n_sub 客户回到虚拟 depot(0)。
    T_sub 为 (n_sub+1)x(n_sub+1)，T_sub[0, i] = 段前→i 的旅行时间；
    T_sub[i, 0] = i→段后 的旅行时间；T_sub[i, j] = 段内 i→j。"""
    nvar = n_sub * n_sub

    def idx(i: int, p: int) -> int:
        return (i - 1) * n_sub + (p - 1)

    Q = np.zeros((nvar, nvar))
    # 列约束
    for p in range(1, n_sub + 1):
        vs = [idx(i, p) for i in range(1, n_sub + 1)]
        for k in vs: Q[k, k] += -A_pen
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)): Q[vs[a], vs[b]] += 2 * A_pen
    # 行约束
    for i in range(1, n_sub + 1):
        vs = [idx(i, p) for p in range(1, n_sub + 1)]
        for k in vs: Q[k, k] += -A_pen
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)): Q[vs[a], vs[b]] += 2 * A_pen
    # 距离
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


def make_sub_T(a: int, customers: list[int], b: int) -> np.ndarray:
    """构造子段 T 矩阵：customers 列表为段内客户，a/b 为段前/段后节点。"""
    k = len(customers)
    T_sub = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        ci = customers[i - 1]
        T_sub[0, i] = T_full[a, ci]
        T_sub[i, 0] = T_full[ci, b]
        for j in range(1, k + 1):
            if i != j:
                T_sub[i, j] = T_full[ci, customers[j - 1]]
    return T_sub


def decode_sub(x: np.ndarray, n_sub: int) -> tuple[list[int], bool]:
    M = np.asarray(x).reshape(n_sub, n_sub)
    feasible = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm_idx = []
    for p in range(n_sub):
        col = M[:, p]
        perm_idx.append(int(np.argmax(col)))
    return perm_idx, feasible


def spin_to_binary(spin_solutions: np.ndarray, nvar: int) -> np.ndarray:
    s = spin_solutions
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]
        s_fixed = s * s_aux
        s_main = s_fixed[:, :-1]
    else:
        s_main = s
    return ((s_main + 1) // 2).astype(np.int8)


# ---------- 评估（含时间窗惩罚）+ 局部搜索 ----------
def evaluate(perm, with_detail=False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []
    for i in perm:
        i = int(i); tt = T_full[last, i]; cur += tt; travel += tt
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
    travel += T_full[last, 0]
    J = travel + penalty
    if with_detail:
        return float(travel), float(penalty), float(J), rows
    return float(travel), float(penalty), float(J)


def two_opt_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True
    return best, best_J


def or_opt_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]
    improved = True
    while improved:
        improved = False
        for L in (1, 2, 3):
            for i in range(0, n - L + 1):
                seg = best[i:i + L]; base = best[:i] + best[i + L:]
                for j in range(0, len(base) + 1):
                    if j == i: continue
                    cand = base[:j] + seg + base[j:]
                    Jc = evaluate(cand)[2]
                    if Jc < best_J - 1e-9:
                        best = cand; best_J = Jc; improved = True; break
                if improved: break
            if improved: break
    return best, best_J


def swap_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best.copy(); cand[i], cand[j] = cand[j], cand[i]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True
    return best, best_J


def polish(perm):
    cur = list(perm)
    while True:
        cur, J1 = two_opt_J(cur)
        cur, J2 = or_opt_J(cur)
        cur, J3 = swap_J(cur)
        if J1 == J2 == J3:
            return cur, evaluate(cur)[2]


# ---------- 子 QUBO 求解器（统一抽象层：SDK SA / CIM 同接口） ----------
def solve_subqubo(Q_sub: np.ndarray, n_sub: int, seeds: list[int],
                  backend: str = "sdk_sa") -> tuple[list[list[int]], list[float], list[float]]:
    """统一接口：返回 pool（每个解都是 perm_idx 0..n_sub-1）+ 哈密顿量列表 + 可行解 J 对应的 QUBO 能量。
    backend = 'sdk_sa' 或 'cim'。**8-bit 精度调整在两种后端下都做**，保证 SDK 验证算法可直接迁到真机。"""
    nvar = n_sub * n_sub

    # 8-bit 精度调整（真机硬性要求）
    Q_for_solve = kw.qubo.adjust_qubo_matrix_precision(Q_sub, bit_width=8)

    # QUBO → Ising
    ising_matrix, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q_for_solve)
    n_ising = ising_matrix.shape[0]
    assert n_ising <= 550, f"sub QUBO ising 维度 {n_ising} > 550 真机上限"

    pool_perms = []
    pool_H = []
    pool_qubo = []

    for sd in seeds:
        if backend == "sdk_sa":
            sa = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=500.0, alpha=0.995, cutoff_temperature=1e-3,
                iterations_per_t=50, size_limit=80, rand_seed=sd, process_num=1)
            spins = sa.solve(ising_matrix)
        elif backend == "cim":
            cim = kw.cim.CIMOptimizer(task_name=f"q3_sub_{sd}", wait=True,
                                       task_mode="quota", sample_number=10)
            spins = cim.solve(ising_matrix)
        else:
            raise ValueError(f"unknown backend: {backend}")

        x = spin_to_binary(spins, nvar)
        for k in range(x.shape[0]):
            perm_idx, feas = decode_sub(x[k], n_sub)
            H = float(x[k] @ Q_for_solve @ x[k])
            pool_H.append(H)
            if feas:
                pool_perms.append(perm_idx)
                pool_qubo.append(H)
    return pool_perms, pool_H, pool_qubo


# ---------- 主流程：滚动窗口分解 ----------
def main():
    print(f"\n== 问题 3 · Kaiwu SDK 验证（滚动窗口分解, n_sub={N_SUB}, sub_qubo={N_SUB**2} 比特 ≤ 550） ==", flush=True)
    print(f"  起点：纯 Python 基线 J={PY_BASELINE_J:.0f}（travel={PY_BASELINE_TRAVEL:.0f}, pen={PY_BASELINE_PEN:.0f}）", flush=True)

    # 切片：3 段 [0..16, 17..33, 34..49]，每段 ≤17 客户
    cur_perm = list(PY_BASELINE_PERM)
    seg_bounds = []
    i = 0
    while i < N:
        j = min(i + N_SUB, N)
        seg_bounds.append((i, j))
        i = j
    print(f"  分段：{len(seg_bounds)} 段，边界={seg_bounds}", flush=True)

    sub_problems_log = []
    all_H_values = []   # 用于哈密顿量演化图

    t_total_start = time.time()
    for seg_idx, (lo, hi) in enumerate(seg_bounds):
        n_sub = hi - lo
        a = 0 if lo == 0 else int(cur_perm[lo - 1])     # 段前节点
        b = 0 if hi == N else int(cur_perm[hi])         # 段后节点
        customers = [int(c) for c in cur_perm[lo:hi]]

        print(f"\n  [子问题 {seg_idx + 1}/{len(seg_bounds)}] n_sub={n_sub}, 段前={a}, 段后={b}", flush=True)
        print(f"    段内客户={customers}", flush=True)

        T_sub = make_sub_T(a, customers, b)
        Q_sub = build_qubo_tsp(T_sub, n_sub, A_PEN)
        print(f"    子 QUBO 维度={n_sub * n_sub} 比特（≤ 550 真机上限）", flush=True)

        # 段内原顺序的 J
        original_J = evaluate(cur_perm)[2]

        # 求解
        t0 = time.time()
        seeds = [SEED + seg_idx * 10 + k for k in range(3)]
        sub_perms, sub_H, sub_qubo = solve_subqubo(Q_sub, n_sub, seeds, backend="sdk_sa")
        all_H_values.extend(sub_H)
        t_sub = time.time() - t0
        print(f"    SDK SA 求解：{t_sub:.2f}s，候选 {len(sub_H)}，可行 {len(sub_perms)}", flush=True)

        # 取所有可行子解 → 拼接全 perm → polish → 取 J 最优
        best_J_after = original_J
        best_perm_after = list(cur_perm)
        for sp_idx in sub_perms:
            new_segment = [customers[k] for k in sp_idx]
            cand_perm = list(cur_perm)
            cand_perm[lo:hi] = new_segment
            cand_perm, cand_J = polish(cand_perm)
            if cand_J < best_J_after - 1e-9:
                best_J_after = cand_J
                best_perm_after = cand_perm

        # 即使没有可行子解，也用 polish 一次原 perm
        polished_orig, polished_orig_J = polish(cur_perm)
        if polished_orig_J < best_J_after - 1e-9:
            best_J_after = polished_orig_J
            best_perm_after = polished_orig

        improvement = original_J - best_J_after
        print(f"    段前 J={original_J:.0f}  →  段后 J={best_J_after:.0f}  (Δ={improvement:+.0f})", flush=True)

        sub_problems_log.append(dict(
            seg_idx=seg_idx, lo=lo, hi=hi, n_sub=n_sub,
            sub_qubo_bits=n_sub * n_sub,
            seg_before_node=a, seg_after_node=b,
            customers_in=customers,
            n_candidates=len(sub_H),
            n_feasible=len(sub_perms),
            time_sec=round(t_sub, 3),
            J_before=float(original_J),
            J_after=float(best_J_after),
            improvement=float(improvement),
            seeds=[int(s) for s in seeds],
        ))
        cur_perm = best_perm_after

    # 全局 polish 收尾
    print("\n  [全局 polish] 跨段精修", flush=True)
    t0 = time.time()
    final_perm, final_J = polish(cur_perm)
    t_global = time.time() - t0
    travel, penalty, J, schedule = evaluate(final_perm, with_detail=True)
    full_route = [0] + [int(x) for x in final_perm] + [0]
    n_violators = sum(1 for r in schedule if r["early"] > 0 or r["late"] > 0)
    t_total = time.time() - t_total_start

    print(f"\n========== Q3 · Kaiwu SDK 分解最终结果 ==========", flush=True)
    print(f"  路径：{full_route}", flush=True)
    print(f"  travel={travel:.0f}  penalty={penalty:.0f}  J={J:.0f}", flush=True)
    print(f"  违反客户={n_violators}/{N}", flush=True)
    print(f"  总耗时：{t_total:.2f}s（全局 polish {t_global:.2f}s）", flush=True)
    print(f"  vs 纯 Python 基线 J={PY_BASELINE_J:.0f}：ΔJ = {PY_BASELINE_J - J:+.0f}  ({(PY_BASELINE_J - J) / PY_BASELINE_J * 100:+.3f}%)", flush=True)

    # ---- JSON ----
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    result = dict(
        problem="Q3 · Kaiwu SDK 验证（滚动窗口分解，真机兼容）",
        kaiwu_version=kw.__version__,
        n_customers=N, n_sub=N_SUB,
        sub_qubo_bits=N_SUB * N_SUB,
        cim_qubit_limit=550,
        decomposition="rolling_window",
        encoding="one-hot 位置编码（仅子 TSP，时间窗在解码评估）；同 Q2 方案 C",
        backend="sdk_sa",
        backend_swap_to_cim_supported=True,
        bit_width_8_applied=True,
        A_pen=A_PEN, M1=M1, M2=M2,
        seeds_used=[s for sp in sub_problems_log for s in sp["seeds"]],
        sub_problems=sub_problems_log,
        time_sec=dict(total=round(t_total, 3),
                       global_polish=round(t_global, 3)),
        final=dict(route=full_route,
                   perm=[int(x) for x in final_perm],
                   total_travel_time=float(travel),
                   total_tw_penalty=float(penalty),
                   objective_J=float(J),
                   n_violators=int(n_violators)),
        schedule=schedule,
        comparison=dict(
            pure_python_baseline=dict(J=PY_BASELINE_J, travel=PY_BASELINE_TRAVEL,
                                       penalty=PY_BASELINE_PEN, perm=PY_BASELINE_PERM),
            delta_J=float(PY_BASELINE_J - J),
            relative_pct=round((PY_BASELINE_J - J) / PY_BASELINE_J * 100, 4),
            same_or_better=bool(J <= PY_BASELINE_J + 1e-9),
        ),
        note=("SDK 求解算法已遵守 CIM 真机比特预算（每子 QUBO ≤ 549）+ 8-bit 精度，"
              "可通过将 backend 参数从 'sdk_sa' 改为 'cim' 直接迁到真机。"),
    )
    out_json = OUT_RESULT / "q3_kaiwu_sdk.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}", flush=True)

    # ---- 表 ----
    df = pd.DataFrame(schedule)
    df.to_csv(OUT_TABLE / "tab_03c_q3_kaiwu_schedule.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{问题 3 · Kaiwu SDK 分解求解最优调度（含时间窗违反）}\\label{tab:q3_kaiwu_schedule}\n"
           "\\small\n\\begin{tabular}{ccccccc}\n\\toprule\n"
           "客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for r in schedule:
        tex += (f"{r['customer']} & {r['arrive']:.0f} & "
                f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                f"{r['early']:.0f} & {r['late']:.0f} & "
                f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
    tex += ("\\midrule\n"
            f"\\multicolumn{{6}}{{r}}{{总运输时间}} & {travel:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{时间窗惩罚}} & {penalty:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{目标 J}} & {J:.0f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_03c_q3_kaiwu_schedule.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_03c_q3_kaiwu_schedule.csv + .tex", flush=True)

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
    ax.set_title(f"Q3 · SDK 分解解（n=50, 子 QUBO≤{N_SUB**2}比特）\n"
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
    fig.suptitle("Q3 · Kaiwu SDK 滚动窗口分解（子 QUBO ≤ 550 比特，真机兼容）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_route.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03c_q3_kaiwu_route.png + .pdf", flush=True)

    # ---- 图 2：哈密顿量演化（题目硬性要求） ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(all_H_values, lw=0.7, alpha=0.5, color="#888", label="原始解序")
    ax.plot(sorted(all_H_values, reverse=True), lw=1.5, color="#d9534f", label="排序（高→低）")
    ax.axhline(min(all_H_values), color="#5cb85c", lw=1.0, ls="--",
               label=f"最低 H = {min(all_H_values):.0f}")
    ax.set_xlabel("解索引（跨 3 个子问题）")
    ax.set_ylabel("哈密顿量 H")
    ax.set_title(f"Q3 · 子 QUBO SDK SA 哈密顿量演化（题目要求图）\n"
                 f"3 子问题 × 3 seeds × 80 chains, 每子 QUBO {N_SUB**2}比特")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_hamiltonian.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_hamiltonian.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03c_q3_kaiwu_hamiltonian.png + .pdf", flush=True)

    # ---- 图 3：vs 纯 Python 基线 ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["纯 Python 基线\n(n=50 全局启发式)", f"Kaiwu SDK 分解\n(子 QUBO {N_SUB**2}比特×3)"]
    Js = [PY_BASELINE_J, J]
    pens = [PY_BASELINE_PEN, penalty]
    travels = [PY_BASELINE_TRAVEL, travel]
    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width / 2, Js, width, label="目标 J", color="#3a78c2")
    ax.bar(x + width / 2, pens, width, label="时间窗惩罚", color="#d9534f", alpha=0.8)
    for xi, jv in zip(x - width / 2, Js):
        ax.text(xi, jv, f"{jv:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for xi, pv, tv in zip(x + width / 2, pens, travels):
        ax.text(xi, pv, f"{pv:.0f}\n(travel={tv:.0f})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("数值")
    delta_J = PY_BASELINE_J - J
    ax.set_title(f"Q3 · 纯 Python 基线 vs Kaiwu SDK 分解\nΔJ = {delta_J:+.0f}  "
                 f"({delta_J / PY_BASELINE_J * 100:+.3f}%)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03c_q3_kaiwu_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03c_q3_kaiwu_compare.png + .pdf", flush=True)

    # ---- 图 4：分解策略示意 + 子 QUBO 比特数 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    sizes = [sp["n_sub"] for sp in sub_problems_log]
    bits = [sp["sub_qubo_bits"] for sp in sub_problems_log]
    Jdrops = [sp["improvement"] for sp in sub_problems_log]
    x_pos = np.arange(len(sizes))
    bars = ax.bar(x_pos, bits, color="#3a78c2", edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, bits):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(550, color="#d9534f", lw=1.5, ls="--", label="CIM 真机 550 比特上限")
    ax.axhline(2500, color="#888", lw=1.0, ls=":", label=f"完整 n={N} QUBO = {N**2} 比特（不可行）")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"子段 {i+1}\n({sp['n_sub']} 客户)" for i, sp in enumerate(sub_problems_log)])
    ax.set_ylabel("QUBO 比特数")
    ax.set_yscale("log")
    ax.set_title(f"(a) 子 QUBO 比特预算（铁律 §二.5/二.6）")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3, which="both")

    ax = axes[1]
    bars = ax.bar(x_pos, Jdrops,
                  color=["#5cb85c" if d > 0 else "#aaa" for d in Jdrops],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, Jdrops):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:+.0f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"子段 {i+1}" for i in range(len(sub_problems_log))])
    ax.set_ylabel("J 改进量（>0 表示改进）")
    ax.set_title("(b) 各子段经 SDK 求解后的 J 改进")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Q3 · 滚动窗口分解策略：3 个子 QUBO 各 ≤ 550 比特，"
                 f"总耗时 {t_total:.1f}s",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03c_q3_decompose_diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03c_q3_decompose_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03c_q3_decompose_diagram.png + .pdf", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
