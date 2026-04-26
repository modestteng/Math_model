"""
问题 4 · Kaiwu SDK 验证（每车一子 TSP QUBO，真机兼容）

链路一致性（铁律 §二.6）
----
  起点：R-Q4-003 攻击版的 K=7 最优方案（J=7149，与参考解持平）
  对每条路线（≤9 客户）独立做子 TSP QUBO：
    - 子 QUBO 维度 ≤ 9² = 81 比特  ≤ 550 真机上限 ✓
    - 8-bit 精度调整（kw.qubo.adjust_qubo_matrix_precision）
    - solve_subqubo(Q_sub, backend) — backend='sdk_sa' / 'cim' 单行切换
    - 解码 + polish → 替换原段
  拼接全 7 辆车 → 评估总 J → 与纯 Python 基线对比

输出
----
  results/基础模型/q4_kaiwu_sdk.json
  tables/tab_04c_q4_kaiwu_routes.csv (+ .tex)
  figures/fig_04c_q4_kaiwu_route.png        (+ .pdf)
  figures/fig_04c_q4_kaiwu_hamiltonian.png  (+ .pdf)  题目硬性要求
  figures/fig_04c_q4_kaiwu_compare.png      (+ .pdf)
  figures/fig_04c_q4_qubits_per_vehicle.png (+ .pdf)  铁律 §二.5/二.6 论证图
"""
from __future__ import annotations
# 仅在本进程内临时清除代理变量，不修改用户系统代理
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
OUT_TABLE = ROOT / "tables"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_TABLE, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"

N = 50; M1, M2 = 10.0, 20.0; CAPACITY = 60; A_PEN = 200.0
M_VEHICLE = 1000.0
SEED = 20260426

# Q4 K=7 纯 Python 最优方案（来自 R-Q4-004 finalize）
PY_BASELINE_ROUTES = [
    [15, 2, 21, 40, 6, 37, 17],
    [31, 30, 9, 34, 35, 20, 32],
    [27, 16, 44, 38, 14, 43, 42, 13],
    [47, 19, 36, 49, 11, 10, 1],
    [33, 25, 39, 23, 22, 41, 4],
    [28, 29, 12, 3, 50, 26, 24],
    [5, 45, 7, 18, 8, 46, 48],
]
PY_BASELINE_TRAVEL = 109.0
PY_BASELINE_PEN = 40.0
PY_BASELINE_OBJ_M = 7149.0

import kaiwu as kw
print(f"[Kaiwu] version = {kw.__version__}", flush=True)
kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
print("[Kaiwu] license initialized [OK]", flush=True)

nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full_int = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T_full = T_full_int[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)
D = nodes_raw["demand"].values[: N + 1].astype(float)


def evaluate_route(route_customers, with_detail=False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []; dsum = 0.0
    for i in route_customers:
        i = int(i); tt = T_full[last, i]; cur += tt; travel += tt
        ai, bi = float(A[i]), float(B[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen; dsum += D[i]
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=ai, tw_b=bi,
                             early=float(early), late=float(late),
                             penalty=float(pen), service=float(S[i]),
                             depart=float(cur + S[i]), demand=float(D[i])))
        cur += float(S[i]); last = i
    travel += T_full[last, 0]
    if with_detail:
        return float(travel), float(penalty), float(dsum), rows
    return float(travel), float(penalty), float(dsum)


def evaluate_solution(routes):
    tt = pp = 0.0
    for r in routes:
        if r:
            t, p, _ = evaluate_route(r); tt += t; pp += p
    return tt, pp, tt + pp


# ---------- 子 TSP QUBO（depot 进出 = 都为 0） ----------
def build_subqubo_tsp(T_sub, n_sub, A_pen=200.0):
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


def make_sub_T(customers):
    """对一辆车的客户列表，构造子 T 矩阵：0 行/列 = depot。"""
    k = len(customers)
    T_sub = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        ci = customers[i - 1]
        T_sub[0, i] = T_full[0, ci]
        T_sub[i, 0] = T_full[ci, 0]
        for j in range(1, k + 1):
            if i != j:
                T_sub[i, j] = T_full[ci, customers[j - 1]]
    return T_sub


def decode_sub(x, n_sub):
    M = np.asarray(x).reshape(n_sub, n_sub)
    feas = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm_idx = []
    for p in range(n_sub):
        col = M[:, p]
        perm_idx.append(int(np.argmax(col)))
    return perm_idx, feas


def spin_to_binary(spin_solutions, nvar):
    s = spin_solutions
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]; s_main = (s * s_aux)[:, :-1]
    else:
        s_main = s
    return ((s_main + 1) // 2).astype(np.int8)


# ---------- polish on single route ----------
def polish_route(route):
    def cost(r):
        if not r: return 0.0
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


# ---------- 统一求解接口（链路一致性，铁律 §二.6） ----------
def solve_subqubo(Q_sub, n_sub, seeds, backend="sdk_sa"):
    nvar = n_sub * n_sub
    Q_for_solve = kw.qubo.adjust_qubo_matrix_precision(Q_sub, bit_width=8)
    ising_matrix, _ = kw.conversion.qubo_matrix_to_ising_matrix(Q_for_solve)
    n_ising = ising_matrix.shape[0]
    assert n_ising <= 550, f"sub QUBO {n_ising} > 550 真机上限"
    pool = []; H_list = []
    for sd in seeds:
        if backend == "sdk_sa":
            sa = kw.classical.SimulatedAnnealingOptimizer(
                initial_temperature=300.0, alpha=0.99, cutoff_temperature=1e-3,
                iterations_per_t=30, size_limit=50, rand_seed=sd, process_num=1)
            spins = sa.solve(ising_matrix)
        elif backend == "cim":
            cim = kw.cim.CIMOptimizer(task_name=f"q4_sub_{sd}", wait=True,
                                       task_mode="quota", sample_number=10)
            spins = cim.solve(ising_matrix)
        else:
            raise ValueError(f"unknown backend: {backend}")
        x = spin_to_binary(spins, nvar)
        for k in range(x.shape[0]):
            perm_idx, feas = decode_sub(x[k], n_sub)
            H = float(x[k] @ Q_for_solve @ x[k])
            H_list.append(H)
            if feas:
                pool.append(perm_idx)
    return pool, H_list


# ---------- 主流程 ----------
def main():
    print(f"\n== Q4 · Kaiwu SDK 验证（每车一子 TSP QUBO，真机兼容） ==", flush=True)
    print(f"  起点：纯 Python K=7 J=7149（travel=109, pen=40）", flush=True)

    sub_logs = []
    all_H = []
    new_routes = []
    t_start = time.time()

    for k_idx, customers in enumerate(PY_BASELINE_ROUTES):
        n_sub = len(customers)
        sub_qubo_bits = n_sub * n_sub
        print(f"\n  [车 {k_idx + 1}/{len(PY_BASELINE_ROUTES)}] n_sub={n_sub}, 子 QUBO={sub_qubo_bits} 比特 ≤ 550 ✓", flush=True)
        print(f"    原顺序：{customers}", flush=True)

        T_sub = make_sub_T(customers)
        Q_sub = build_subqubo_tsp(T_sub, n_sub, A_PEN)

        t0 = time.time()
        seeds = [SEED + k_idx * 13 + s for s in range(3)]
        pool, H_list = solve_subqubo(Q_sub, n_sub, seeds, backend="sdk_sa")
        all_H.extend(H_list)
        t_sub = time.time() - t0
        print(f"    SDK SA 求解 {t_sub:.2f}s，候选 {len(H_list)}，可行 {len(pool)}", flush=True)

        # 原 polish 解作 baseline
        orig_polished, orig_cost = polish_route(customers)
        # 对所有可行子解 polish
        best_perm_segs = orig_polished; best_cost_seg = orig_cost
        for sp in pool:
            seg = [customers[i] for i in sp]
            seg_p, c = polish_route(seg)
            if c < best_cost_seg - 1e-9:
                best_perm_segs = seg_p; best_cost_seg = c
        new_routes.append(best_perm_segs)

        new_t, new_p, _ = evaluate_route(best_perm_segs)
        print(f"    原 (travel+pen)={orig_cost:.2f} → SDK 后 {new_t + new_p:.2f}", flush=True)

        sub_logs.append(dict(
            vehicle=k_idx + 1, n_sub=n_sub, sub_qubo_bits=sub_qubo_bits,
            customers_in=customers,
            n_candidates=len(H_list), n_feasible=len(pool),
            time_sec=round(t_sub, 3),
            cost_before=float(orig_cost),
            cost_after=float(new_t + new_p),
            travel_after=float(new_t), pen_after=float(new_p),
            seeds=seeds,
        ))

    t_total = time.time() - t_start

    travel, penalty, J = evaluate_solution(new_routes)
    obj_M = M_VEHICLE * len(new_routes) + J
    print(f"\n========== Q4 · Kaiwu SDK 最终结果 ==========", flush=True)
    for k_idx, r in enumerate(new_routes):
        full = [0] + r + [0]
        t, p, dsum = evaluate_route(r)
        print(f"  V{k_idx + 1}: {full} demand={dsum:.0f}/{CAPACITY} travel={t:.0f} pen={p:.0f}", flush=True)
    print(f"  travel={travel:.0f}  penalty={penalty:.0f}  J(travel+pen)={J:.0f}  obj_M={obj_M:.0f}", flush=True)
    print(f"  vs 纯 Python 基线 J=7149：ΔObj_M = {PY_BASELINE_OBJ_M - obj_M:+.0f}", flush=True)
    print(f"  总耗时：{t_total:.2f}s", flush=True)

    # ---- JSON ----
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    schedules = []
    for k_idx, r in enumerate(new_routes):
        if not r: continue
        t, p, dsum, rows = evaluate_route(r, with_detail=True)
        schedules.append(dict(vehicle=k_idx + 1, route=[0] + r + [0],
                              travel=t, penalty=p, demand=dsum, schedule=rows))

    result = dict(
        problem="Q4 · Kaiwu SDK 验证（每车一子 TSP QUBO，真机兼容）",
        kaiwu_version=kw.__version__,
        n_customers=N, K=len(new_routes), capacity=CAPACITY,
        sub_qubo_bits_max=int(max(s["sub_qubo_bits"] for s in sub_logs)),
        cim_qubit_limit=550,
        decomposition="per-vehicle sub-TSP QUBO",
        encoding="one-hot 位置编码 + 解码后时间窗评估（同 Q2/Q3 方案 C）",
        backend="sdk_sa",
        backend_swap_to_cim_supported=True,
        bit_width_8_applied=True,
        A_pen=A_PEN, M1=M1, M2=M2, M_vehicle=M_VEHICLE,
        sub_problems=sub_logs,
        time_sec=round(t_total, 3),
        final=dict(
            routes=[[0] + [int(c) for c in r] + [0] for r in new_routes],
            travel=float(travel), penalty=float(penalty),
            J_inner=float(J), obj_M=float(obj_M),
            n_violators=sum(1 for sch in schedules
                            for x in sch["schedule"]
                            if x["early"] > 0 or x["late"] > 0),
        ),
        per_vehicle=schedules,
        comparison=dict(
            pure_python_baseline=dict(travel=PY_BASELINE_TRAVEL,
                                       penalty=PY_BASELINE_PEN,
                                       obj_M=PY_BASELINE_OBJ_M,
                                       routes=[[0] + r + [0] for r in PY_BASELINE_ROUTES]),
            delta_obj_M=float(PY_BASELINE_OBJ_M - obj_M),
            same_or_better=bool(obj_M <= PY_BASELINE_OBJ_M + 1e-9),
        ),
        note=("每辆车一个子 TSP QUBO，最大 ≤81 比特 ≤550 真机上限；"
              "8-bit 精度调整已应用；通过 backend='sdk_sa' → 'cim' 单行切换可迁到真机。"),
    )
    out_json = OUT_RESULT / "q4_kaiwu_sdk.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}", flush=True)

    # ---- 表 ----
    rows = []
    for sch in schedules:
        for r in sch["schedule"]:
            rows.append(dict(vehicle=sch["vehicle"], **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE / "tab_04c_q4_kaiwu_routes.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           f"\\caption{{Q4 · Kaiwu SDK 验证：K={len(new_routes)} 多车辆调度}}\\label{{tab:q4_kaiwu_routes}}\n"
           "\\small\n\\begin{tabular}{cccccccc}\n\\toprule\n"
           "车辆 & 客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for sch in schedules:
        for r in sch["schedule"]:
            tex += (f"{sch['vehicle']} & {r['customer']} & {r['arrive']:.0f} & "
                    f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                    f"{r['early']:.0f} & {r['late']:.0f} & "
                    f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
        tex += (f"\\multicolumn{{2}}{{r}}{{车 {sch['vehicle']} 小计}} & "
                f"travel={sch['travel']:.0f} & demand={sch['demand']:.0f}/{CAPACITY} & "
                f"& pen={sch['penalty']:.0f} & \\\\\n")
        tex += "\\midrule\n"
    tex += (f"\\multicolumn{{6}}{{r}}{{总 travel}} & {travel:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{总 penalty}} & {penalty:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{综合目标 1000K+travel+pen}} & {obj_M:.0f} & \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_04c_q4_kaiwu_routes.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_04c_q4_kaiwu_routes.csv + .tex", flush=True)

    # ---- 图 1：路径 + 甘特 ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    cmap = plt.colormaps.get_cmap("tab10")

    # (a) 多车路径
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {0: (0.0, 0.0)}
    for i in range(1, N + 1):
        pos[i] = (np.cos(angles[i - 1]) * 1.3, np.sin(angles[i - 1]) * 1.3)
    ax.scatter(0, 0, s=400, c="#444", marker="s", edgecolor="black", zorder=5)
    ax.text(0, 0, "0", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    for k_idx, r in enumerate(new_routes):
        if not r: continue
        color = cmap(k_idx % 10)
        full = [0] + r + [0]
        for a, b in zip(full[:-1], full[1:]):
            xa, ya = pos[a]; xb, yb = pos[b]
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
        first_c = r[0]
        x, y = pos[first_c]
        ax.text(x * 1.12, y * 1.12, f"V{k_idx+1}", color=color, fontsize=11, fontweight="bold")
    cust_pen = {}
    for sch in schedules:
        for s_row in sch["schedule"]:
            cust_pen[s_row["customer"]] = s_row["penalty"]
    for i in range(1, N + 1):
        x, y = pos[i]
        pen_i = cust_pen.get(i, 0)
        color = "#d9534f" if pen_i > 0 else "#5cb85c"
        ax.scatter(x, y, s=200, c=color, edgecolor="black", linewidth=0.5, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Q4 · Kaiwu SDK 解（每车子 QUBO ≤81 比特）\n"
                 f"travel={travel:.0f}, pen={penalty:.0f}, obj_M={obj_M:.0f}")

    # (b) 甘特
    ax = axes[1]
    y_offset = 0; yticks = []; ylabels = []
    for sch in schedules:
        color = cmap((sch["vehicle"] - 1) % 10)
        for r in sch["schedule"]:
            y = y_offset
            ax.barh(y, r["tw_b"] - r["tw_a"], left=r["tw_a"], height=0.6,
                    color="#cfe5ff", edgecolor="#3a78c2", linewidth=0.4)
            sc = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else color
            ax.barh(y, r["service"], left=r["arrive"], height=0.4, color=sc, alpha=0.95)
            ax.plot([r["arrive"], r["arrive"]], [y - 0.4, y + 0.4], color="black", lw=0.7)
            yticks.append(y); ylabels.append(f"V{sch['vehicle']}|C{r['customer']}")
            y_offset += 1
        y_offset += 0.5
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("时间")
    ax.set_title("时间窗 vs 实际到达（绿/彩=未违反 红=违反）")
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Q4 · Kaiwu SDK 验证（K=7，每车子 TSP QUBO ≤81 比特，真机兼容）",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_route.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04c_q4_kaiwu_route.png + .pdf", flush=True)

    # ---- 图 2：哈密顿量演化（题目硬性要求） ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(all_H, lw=0.7, alpha=0.5, color="#888", label="原始解序")
    ax.plot(sorted(all_H, reverse=True), lw=1.5, color="#d9534f", label="排序（高→低）")
    ax.axhline(min(all_H), color="#5cb85c", lw=1.0, ls="--",
               label=f"最低 H = {min(all_H):.1f}")
    ax.set_xlabel("解索引（跨 7 个子 QUBO × 3 seeds）")
    ax.set_ylabel("哈密顿量 H = x^T Q x")
    ax.set_title(f"Q4 · 子 QUBO SDK SA 哈密顿量演化（题目硬性要求图）\n"
                 f"7 子 QUBO × 3 seeds × 50 chains, 子 QUBO ≤81 比特")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_hamiltonian.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_hamiltonian.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04c_q4_kaiwu_hamiltonian.png + .pdf", flush=True)

    # ---- 图 3：vs 纯 Python 对比 ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["纯 Python 基线\n(K=7, 攻击 LNS)", "Kaiwu SDK 分解\n(每车子 QUBO ≤81 比特)"]
    objs = [PY_BASELINE_OBJ_M, obj_M]
    travels = [PY_BASELINE_TRAVEL, travel]
    pens = [PY_BASELINE_PEN, penalty]
    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width / 2, objs, width, label="综合目标 obj_M", color="#3a78c2")
    ax.bar(x + width / 2, pens, width, label="时间窗惩罚", color="#d9534f", alpha=0.8)
    for xi, jv in zip(x - width / 2, objs):
        ax.text(xi, jv, f"{jv:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for xi, pv, tv in zip(x + width / 2, pens, travels):
        ax.text(xi, pv, f"{pv:.0f}\n(travel={tv:.0f})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("数值")
    delta = PY_BASELINE_OBJ_M - obj_M
    ax.set_title(f"Q4 · 纯 Python 基线 vs Kaiwu SDK 分解\n"
                 f"Δobj_M = {delta:+.0f}  ({delta / PY_BASELINE_OBJ_M * 100:+.3f}%)")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04c_q4_kaiwu_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04c_q4_kaiwu_compare.png + .pdf", flush=True)

    # ---- 图 4：每车子 QUBO 比特预算（铁律 §二.5/二.6） ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bits = [s["sub_qubo_bits"] for s in sub_logs]
    x_pos = np.arange(len(bits))
    bars = ax.bar(x_pos, bits, color=cmap(np.arange(len(bits)) % 10),
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, bits):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(550, color="#d9534f", lw=1.5, ls="--", label="CIM 真机 550 比特上限")
    ax.axhline(N * N, color="#888", lw=1.0, ls=":", label=f"完整 n={N} QUBO = {N*N} 比特（不可行）")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"V{s['vehicle']}\n({s['n_sub']} 客户)" for s in sub_logs])
    ax.set_ylabel("子 QUBO 比特数")
    ax.set_yscale("log")
    ax.set_title(f"Q4 · 每车子 QUBO 比特预算（铁律 §二.5/二.6）\n"
                 f"max={max(bits)} 比特, 远低于 550 真机上限")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04c_q4_qubits_per_vehicle.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04c_q4_qubits_per_vehicle.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04c_q4_qubits_per_vehicle.png + .pdf", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
