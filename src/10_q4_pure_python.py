"""
问题 4 · 第 1 步：多车辆 VRP（容量 + 时间窗）纯 Python 基线 + K 灵敏度

模型
----
  设有 K 辆容量相同（=60）的车，从 depot=0 出发回到 depot=0。
  客户 i ∈ {1..50} 各自有 demand[i]、时间窗 [a_i, b_i]、服务时间 s_i=2。
  每辆车 k 走一条路线 R_k = (depot → 客户序列 → depot)，必须满足 ∑demand ≤ 60。
  无空闲等待：t_i = arrival time at i。
  时间窗罚 P_i = M1·([a_i-t_i]^+)² + M2·([t_i-b_i]^+)²，M1=10, M2=20。

  目标 J(K) = sum_k (travel_k + tw_penalty_k)，对每个 K 单独优化。

求解
----
  1) 启发式分配：Clarke-Wright Savings 算法（容量约束下合并客户）→ 给定 K 辆车的初始 K 条路线；
     若 Savings 输出 != K 条，按 J 贪心合并/分裂到目标 K。
  2) per-vehicle polish：2-opt + Or-opt + swap 反复轮转，按 J 评估。
  3) 跨车 swap / move：考虑把客户从车 a 移到车 b（容量允许时）→ J 改进则接受。
  4) 对每个 K ∈ [5, 8] 重复 1-3，输出 K 灵敏度。

输出
----
  results/基础模型/q4_pure_python.json
  results/灵敏度分析/q4_K_sensitivity.json
  tables/tab_04_q4_routes.csv (+ .tex)
  tables/tab_04_q4_K_sensitivity.csv (+ .tex)
  figures/fig_04_q4_routes.png        (+ .pdf)  多车路径图
  figures/fig_04_q4_gantt.png         (+ .pdf)  多车甘特
  figures/fig_04_q4_K_sensitivity.png (+ .pdf)  K vs J/travel/penalty 三曲线
  figures/fig_04_q4_violation.png     (+ .pdf)  50 客户违反分布
"""
from __future__ import annotations
import json
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT_RESULT = ROOT / "results/基础模型"
OUT_SENS = ROOT / "results/灵敏度分析"
OUT_TABLE = ROOT / "tables"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_SENS, OUT_TABLE, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

N = 50
M1, M2 = 10.0, 20.0
CAPACITY = 60
K_RANGE = [5, 6, 7, 8]
SEED = 20260426
LNS_ITERS = 200          # 每 K 的 LNS 轮数
SA_SEEDS = 2             # 每 K 的 SA 起点数

# ---- 数据 ----
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_full[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)
D = nodes_raw["demand"].values[: N + 1].astype(float)
CAP_FROM_DATA = float(nodes_raw["capacity"].values[0])
assert CAP_FROM_DATA == CAPACITY, f"data capacity {CAP_FROM_DATA} != {CAPACITY}"
TOTAL_DEMAND = float(D[1:N + 1].sum())
K_MIN = int(np.ceil(TOTAL_DEMAND / CAPACITY))
print(f"[Data] n={N}  total_demand={TOTAL_DEMAND}  capacity={CAPACITY}  K_min=⌈{TOTAL_DEMAND}/{CAPACITY}⌉={K_MIN}", flush=True)


# ---------- 单条路线评估（含时间窗罚 + 容量校验） ----------
def evaluate_route(route_customers, with_detail=False):
    """route_customers: 不含 depot 的客户列表。返回 (travel, penalty, demand_sum, rows)."""
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []; dsum = 0.0
    for i in route_customers:
        i = int(i)
        tt = T[last, i]; cur += tt; travel += tt
        ai, bi = float(A[i]), float(B[i])
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen
        dsum += D[i]
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=ai, tw_b=bi,
                             early=float(early), late=float(late),
                             penalty=float(pen), service=float(S[i]),
                             depart=float(cur + S[i]), demand=float(D[i])))
        cur += float(S[i]); last = i
    travel += T[last, 0]
    if with_detail:
        return float(travel), float(penalty), float(dsum), rows
    return float(travel), float(penalty), float(dsum)


def evaluate_solution(routes):
    """routes: list of route (each = list of customer ids without depot)."""
    total_travel = 0.0; total_pen = 0.0; per_vehicle = []
    for r in routes:
        if not r:
            per_vehicle.append((0.0, 0.0, 0.0)); continue
        tr, pn, dsum = evaluate_route(r)
        total_travel += tr; total_pen += pn
        per_vehicle.append((tr, pn, dsum))
    J = total_travel + total_pen
    return total_travel, total_pen, J, per_vehicle


def feasible_capacity(routes):
    return all(sum(D[i] for i in r) <= CAPACITY for r in routes)


# ---------- Clarke-Wright Savings ----------
def clarke_wright_savings():
    """返回 K 条路线（K 由 savings 自动决定）。"""
    routes = [[i] for i in range(1, N + 1)]
    in_route = {i: idx for idx, i in enumerate(range(1, N + 1))}
    savings = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = T[0, i] + T[0, j] - T[i, j]
            savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])

    def route_demand(r):
        return sum(D[c] for c in r)

    for s, i, j in savings:
        ri = in_route.get(i); rj = in_route.get(j)
        if ri is None or rj is None or ri == rj:
            continue
        # i must be at one end, j at one end
        Ri = routes[ri]; Rj = routes[rj]
        if route_demand(Ri) + route_demand(Rj) > CAPACITY:
            continue
        # 4 种合并方式（i 在 Ri 头/尾，j 在 Rj 头/尾）
        merged = None
        if Ri[-1] == i and Rj[0] == j:
            merged = Ri + Rj
        elif Ri[0] == i and Rj[-1] == j:
            merged = Rj + Ri
        elif Ri[-1] == i and Rj[-1] == j:
            merged = Ri + Rj[::-1]
        elif Ri[0] == i and Rj[0] == j:
            merged = Ri[::-1] + Rj
        else:
            continue
        routes[ri] = merged
        for c in Rj:
            in_route[c] = ri
        routes[rj] = []
    final = [r for r in routes if r]
    return final


def adjust_to_K(routes, target_K, rng):
    """把 routes 调整到恰好 K 条（合并最小或拆分最大）。"""
    routes = [list(r) for r in routes if r]
    while len(routes) > target_K:
        # 合并 demand 之和最小的两条（容量允许时）
        # 找一对总 demand 最小的可合并组合
        best = None
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                ds = sum(D[c] for c in routes[i]) + sum(D[c] for c in routes[j])
                if ds <= CAPACITY:
                    if best is None or ds < best[0]:
                        best = (ds, i, j)
        if best is None:
            break  # 容量不允许，提前退出
        _, i, j = best
        routes[i] = routes[i] + routes[j]
        routes.pop(j)
    while len(routes) < target_K:
        # 把最大 demand 路线拆成两半
        idx = max(range(len(routes)), key=lambda k: sum(D[c] for c in routes[k]))
        r = routes[idx]
        if len(r) < 2:
            break
        mid = len(r) // 2
        routes[idx] = r[:mid]
        routes.append(r[mid:])
    return routes


# ---------- per-vehicle polish ----------
def polish_route(route):
    """对单条路线做 2-opt + Or-opt + swap，按单车 (travel + pen) 评估。"""
    def cost(r):
        if not r: return 0.0
        tr, pn, _ = evaluate_route(r)
        return tr + pn

    cur = list(route); cur_cost = cost(cur)
    n = len(cur)
    if n <= 1: return cur, cur_cost

    improved = True
    while improved:
        improved = False
        # 2-opt
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = cur[:i] + cur[i:k + 1][::-1] + cur[k + 1:]
                c = cost(cand)
                if c < cur_cost - 1e-9:
                    cur, cur_cost = cand, c; improved = True
        # Or-opt L=1,2,3
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
        # swap
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = cur.copy(); cand[i], cand[j] = cand[j], cand[i]
                c = cost(cand)
                if c < cur_cost - 1e-9:
                    cur, cur_cost = cand, c; improved = True
    return cur, cur_cost


def polish_all_vehicles(routes):
    return [polish_route(r)[0] if r else r for r in routes]


# ---------- 跨车 move/swap ----------
def cross_vehicle_optimize(routes, max_iter=10):
    """每轮：枚举把客户 c 从车 a 移到车 b 的所有组合（快速评估，不 polish），
    + 跨车 swap：交换两辆车上的两个客户（容量允许时）。"""
    routes = [list(r) for r in routes]
    _, _, best_J, _ = evaluate_solution(routes)

    for _ in range(max_iter):
        improved = False
        # ----- 1) move -----
        for a in range(len(routes)):
            if improved: break
            for ci_idx, c in enumerate(list(routes[a])):
                if improved: break
                for b in range(len(routes)):
                    if b == a: continue
                    if sum(D[x] for x in routes[b]) + D[c] > CAPACITY: continue
                    base_a = routes[a][:ci_idx] + routes[a][ci_idx + 1:]
                    for pos in range(len(routes[b]) + 1):
                        new_b = routes[b][:pos] + [c] + routes[b][pos:]
                        new_routes = list(routes)
                        new_routes[a] = base_a; new_routes[b] = new_b
                        _, _, new_J, _ = evaluate_solution(new_routes)
                        if new_J < best_J - 1e-9:
                            routes = [list(r) for r in new_routes]
                            best_J = new_J; improved = True; break
                    if improved: break
        if improved: continue
        # ----- 2) swap -----
        for a in range(len(routes)):
            if improved: break
            for b in range(a + 1, len(routes)):
                if improved: break
                for ia, ca in enumerate(list(routes[a])):
                    if improved: break
                    for ib, cb in enumerate(list(routes[b])):
                        # 容量校验
                        new_da = sum(D[x] for x in routes[a]) - D[ca] + D[cb]
                        new_db = sum(D[x] for x in routes[b]) - D[cb] + D[ca]
                        if new_da > CAPACITY or new_db > CAPACITY: continue
                        new_a = routes[a][:ia] + [cb] + routes[a][ia + 1:]
                        new_b = routes[b][:ib] + [ca] + routes[b][ib + 1:]
                        new_routes = list(routes)
                        new_routes[a] = new_a; new_routes[b] = new_b
                        _, _, new_J, _ = evaluate_solution(new_routes)
                        if new_J < best_J - 1e-9:
                            routes = [list(r) for r in new_routes]
                            best_J = new_J; improved = True; break
        if not improved: break
    return routes, best_J


# ---------- 多种 warm start ----------
def nn_with_capacity_start(K_target, rng, weights=(0.5, 0.5)):
    """NN with capacity：从 depot 开始按"travel + 时间窗紧迫"贪心分配，
    超容量则开新车。返回路线列表（K 由生成决定）。"""
    w_t, w_late = weights
    unv = set(range(1, N + 1))
    routes = []
    cur_route = []
    cur = 0; cur_t = 0; cur_d = 0
    while unv:
        # 候选：所有还没走的且不超容量的
        feasible = [j for j in unv if cur_d + D[j] <= CAPACITY]
        if not feasible:
            # 关闭当前路线，开新车
            routes.append(cur_route); cur_route = []
            cur = 0; cur_t = 0; cur_d = 0
            continue
        def score(j):
            arr = cur_t + T[cur, j]
            late = max(0.0, arr - B[j])
            return w_t * T[cur, j] + w_late * late
        nxt = min(feasible, key=score)
        cur_route.append(nxt); unv.discard(nxt)
        cur_t += T[cur, nxt] + S[nxt]; cur = nxt; cur_d += D[nxt]
    if cur_route: routes.append(cur_route)
    return routes


def time_sweep_start(K_target):
    """按时间窗中位数排序后切片成 K 段，每段填到容量上限。"""
    customers_sorted = sorted(range(1, N + 1), key=lambda i: (A[i] + B[i]) / 2)
    routes = []; cur_route = []; cur_d = 0
    for c in customers_sorted:
        if cur_d + D[c] > CAPACITY:
            routes.append(cur_route); cur_route = [c]; cur_d = D[c]
        else:
            cur_route.append(c); cur_d += D[c]
    if cur_route: routes.append(cur_route)
    return routes


def collect_warm_starts(K_target, rng):
    """对目标 K 收集多种 warm start，所有都通过 adjust_to_K 调到 K。"""
    candidates = []
    # Savings
    candidates.append(("savings", clarke_wright_savings()))
    # NN with capacity 4 种权重
    for w in [(1.0, 0.0), (1.0, 0.5), (0.5, 1.0), (0.3, 2.0)]:
        candidates.append((f"nn_cap_{w}", nn_with_capacity_start(K_target, rng, w)))
    # Time sweep
    candidates.append(("time_sweep", time_sweep_start(K_target)))
    # 调整到 K + 容量校验
    out = []
    for name, routes in candidates:
        adjusted = adjust_to_K(routes, K_target, rng)
        if feasible_capacity(adjusted) and len(adjusted) == K_target:
            out.append((name, adjusted))
    return out


# ---------- LNS（跨车 ruin-and-recreate） ----------
def regret2_cross_insert(routes, removed):
    """把 removed 客户按 regret-2 顺序插到 routes 最佳位置（容量允许时）。"""
    routes = [list(r) for r in routes]
    rem = list(removed)
    while rem:
        best_choice = None  # (regret_score, c, b, pos)
        for c in rem:
            costs = []  # (J 增量, b, pos)
            for b in range(len(routes)):
                if sum(D[x] for x in routes[b]) + D[c] > CAPACITY:
                    continue
                _, _, J_old, _ = evaluate_solution(routes)
                for pos in range(len(routes[b]) + 1):
                    new_routes = [list(r) for r in routes]
                    new_routes[b] = routes[b][:pos] + [c] + routes[b][pos:]
                    _, _, J_new, _ = evaluate_solution(new_routes)
                    costs.append((J_new - J_old, b, pos))
            if not costs:
                # 无可插位置（容量满），强制插到当前最小路线（破坏容量）—— 这里跳过
                continue
            costs.sort()
            best_J = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else best_J
            regret = second - best_J
            score = -regret + best_J * 1e-3
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, c, costs[0][1], costs[0][2])
        if best_choice is None:
            # 所有 c 都无可插位置，强插到 demand 最小路线（违反容量，但保证完整覆盖）
            c = rem[0]
            b = min(range(len(routes)), key=lambda k: sum(D[x] for x in routes[k]))
            new_routes = [list(r) for r in routes]
            new_routes[b] = routes[b] + [c]
            routes = new_routes; rem.remove(c)
            continue
        _, c, b, pos = best_choice
        routes[b] = routes[b][:pos] + [c] + routes[b][pos:]
        rem.remove(c)
    return routes


def lns_cross_vehicle(routes, rng, n_iter=200, ruin_min=3, ruin_max=6,
                     T0=200.0, T_decay=0.995):
    """大邻域搜索：每轮摧毁 ruin_min..ruin_max 个客户跨车重插。"""
    cur_routes = [list(r) for r in routes]
    _, _, cur_J, _ = evaluate_solution(cur_routes)
    best_routes = [list(r) for r in cur_routes]; best_J = cur_J
    history = [best_J]
    T_now = T0

    for it in range(n_iter):
        k = int(rng.integers(ruin_min, ruin_max + 1))
        all_customers = [c for r in cur_routes for c in r]
        if rng.random() < 0.5:
            # 按罚高优先摧毁
            cust_pen = []
            for r_idx, r in enumerate(cur_routes):
                if not r: continue
                _, _, _, rows = evaluate_route(r, with_detail=True)
                for x in rows:
                    cust_pen.append((x["customer"], x["penalty"]))
            cust_pen.sort(key=lambda x: -x[1])
            top_pool = [c for c, _ in cust_pen[:max(k * 2, 8)]]
            removed = list(rng.choice(top_pool, size=min(k, len(top_pool)), replace=False))
        else:
            removed = list(rng.choice(all_customers, size=min(k, len(all_customers)), replace=False))
        # 移除
        partial = [[c for c in r if c not in removed] for r in cur_routes]
        # regret-2 重插
        cand = regret2_cross_insert(partial, removed)
        # 容量违规则丢弃
        if not feasible_capacity(cand):
            history.append(best_J); T_now *= T_decay; continue
        # 短 polish
        cand = polish_all_vehicles(cand)
        _, _, cand_J, _ = evaluate_solution(cand)
        dJ = cand_J - cur_J
        if dJ < 0 or rng.random() < np.exp(-dJ / max(T_now, 1e-6)):
            cur_routes = cand; cur_J = cand_J
            if cur_J < best_J:
                best_routes = [list(r) for r in cur_routes]; best_J = cur_J
        history.append(best_J)
        T_now *= T_decay
    return best_routes, best_J, history


# ---------- 主流程：每 K 多起点 + LNS + 多 seed ----------
def solve_for_K(target_K, rng_master):
    """对给定 K 求解：多 warm start + per-vehicle polish + 跨车 → LNS（多 seed）→ 收尾 polish。"""
    starts = collect_warm_starts(target_K, rng_master)
    if not starts:
        print(f"  [K={target_K}] 无可行 warm start，跳过", flush=True)
        return None
    print(f"  [K={target_K}] {len(starts)} 个 warm start", flush=True)

    # 阶段 1：每个 warm start 做 polish + 跨车 → 取 J 最优
    candidates = []
    for name, routes in starts:
        routes_p = polish_all_vehicles(routes)
        routes_p, _ = cross_vehicle_optimize(routes_p, max_iter=5)
        routes_p = polish_all_vehicles(routes_p)
        _, _, J, _ = evaluate_solution(routes_p)
        candidates.append((J, routes_p, name))
    candidates.sort(key=lambda x: x[0])
    base_J, base_routes, base_name = candidates[0]
    print(f"  [K={target_K}] 多起点 polish: best J={base_J:.0f} (来源={base_name})", flush=True)
    print(f"    TOP-3 J = {[round(c[0]) for c in candidates[:3]]}", flush=True)

    # 阶段 2：LNS × SA_SEEDS 个种子
    overall_best_J = base_J
    overall_best_routes = [list(r) for r in base_routes]
    lns_history_best = None
    for sd_idx in range(SA_SEEDS):
        rng_lns = np.random.default_rng(SEED + target_K * 100 + sd_idx)
        # 起点：交替用 base 和 TOP-2/3 候选
        seed_start_idx = sd_idx % min(len(candidates), 3)
        start_routes = [list(r) for r in candidates[seed_start_idx][1]]
        lns_best, lns_J, lns_hist = lns_cross_vehicle(
            start_routes, rng_lns, n_iter=LNS_ITERS, ruin_min=3, ruin_max=6
        )
        # 收尾 polish + cross
        lns_best = polish_all_vehicles(lns_best)
        lns_best, _ = cross_vehicle_optimize(lns_best, max_iter=5)
        lns_best = polish_all_vehicles(lns_best)
        _, _, lns_polished_J, _ = evaluate_solution(lns_best)
        print(f"    [K={target_K}] LNS seed#{sd_idx}: J(LNS)={lns_J:.0f} → polish={lns_polished_J:.0f}", flush=True)
        if lns_polished_J < overall_best_J - 1e-9:
            overall_best_J = lns_polished_J
            overall_best_routes = [list(r) for r in lns_best]
            lns_history_best = lns_hist

    # 收尾
    routes = overall_best_routes
    travel, pen, J, per_v = evaluate_solution(routes)
    print(f"  [K={target_K}] 最终 J={J:.0f} (travel={travel:.0f}, pen={pen:.0f})", flush=True)

    return dict(K=target_K, routes=routes, travel=float(travel),
                penalty=float(pen), J=float(J), per_vehicle=per_v,
                lns_history=lns_history_best)


def main():
    print(f"\n== Q4 · 多车辆 VRP（容量 + 时间窗）纯 Python ==", flush=True)
    rng = np.random.default_rng(SEED)

    K_results = {}
    t0 = time.time()
    for K in K_RANGE:
        print(f"\n[K = {K}] ----------------------------------------", flush=True)
        res = solve_for_K(K, rng)
        if res is None:
            print(f"  [K={K}] 容量不可行，跳过", flush=True)
            continue
        K_results[K] = res
    t_total = time.time() - t0
    print(f"\n[All K] 总耗时 {t_total:.1f}s", flush=True)

    # 选 K* = J 最小者
    best_K = min(K_results.keys(), key=lambda k: K_results[k]["J"])
    best_res = K_results[best_K]
    print(f"\n========== Q4 最优 K* = {best_K} ==========", flush=True)
    print(f"  J = {best_res['J']:.0f}  travel = {best_res['travel']:.0f}  penalty = {best_res['penalty']:.0f}", flush=True)
    for k_idx, r in enumerate(best_res["routes"]):
        full = [0] + r + [0]
        dsum = sum(D[c] for c in r)
        tr, pn, _ = evaluate_route(r)
        print(f"  车 {k_idx+1}: 路径={full}  demand={dsum:.0f}/{CAPACITY}  travel={tr:.0f}  pen={pn:.0f}", flush=True)

    # ---- 全 50 客户调度（每车独立，时间从 0 起算） ----
    schedule_per_vehicle = []
    n_violators_total = 0
    for k_idx, r in enumerate(best_res["routes"]):
        if not r: continue
        tr, pn, dsum, rows = evaluate_route(r, with_detail=True)
        schedule_per_vehicle.append(dict(vehicle=k_idx + 1, route=[0] + r + [0],
                                          travel=tr, penalty=pn, demand=dsum,
                                          schedule=rows))
        n_violators_total += sum(1 for x in rows if x["early"] > 0 or x["late"] > 0)

    # ---- JSON ----
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    result = dict(
        problem="Q4 · 多车辆 VRP（容量 + 时间窗）纯 Python 基线",
        n_customers=N, M1=M1, M2=M2, capacity=CAPACITY,
        total_demand=TOTAL_DEMAND, K_min=K_MIN,
        K_range=K_RANGE,
        seed=SEED,
        time_sec=round(t_total, 2),
        best_K=best_K,
        best=dict(
            J=best_res["J"], travel=best_res["travel"], penalty=best_res["penalty"],
            routes=[[0] + [int(c) for c in r] + [0] for r in best_res["routes"]],
            n_violators=n_violators_total,
        ),
        per_vehicle=schedule_per_vehicle,
        K_sensitivity={int(k): dict(J=v["J"], travel=v["travel"], penalty=v["penalty"],
                                    n_routes=sum(1 for r in v["routes"] if r),
                                    routes=[[0] + [int(c) for c in r] + [0] for r in v["routes"]])
                       for k, v in K_results.items()},
        note="启发式分配（Savings + 调整到 K）+ per-vehicle polish + 跨车 move 优化",
    )
    out_json = OUT_RESULT / "q4_pure_python.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}", flush=True)

    # K 灵敏度单独一份
    sens_json = OUT_SENS / "q4_K_sensitivity.json"
    sens_data = dict(
        K_range=K_RANGE,
        results={int(k): dict(J=v["J"], travel=v["travel"], penalty=v["penalty"],
                              n_routes=sum(1 for r in v["routes"] if r))
                 for k, v in K_results.items()},
        best_K=best_K,
        note="J = travel + tw_penalty 之和；K 取最小值即为最优配置",
    )
    sens_json.write_text(json.dumps(sens_data, ensure_ascii=False, indent=2, default=_to_jsonable),
                          encoding="utf-8")
    print(f"[写出] {sens_json.relative_to(ROOT)}", flush=True)

    # ---- 路线表 ----
    rows = []
    for v in schedule_per_vehicle:
        for r in v["schedule"]:
            rows.append(dict(vehicle=v["vehicle"], **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE / "tab_04_q4_routes.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           f"\\caption{{问题 4 最优 K^*={best_K} 多车辆调度}}\\label{{tab:q4_routes}}\n"
           "\\small\n\\begin{tabular}{cccccccc}\n\\toprule\n"
           "车辆 & 客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for v in schedule_per_vehicle:
        for r in v["schedule"]:
            tex += (f"{v['vehicle']} & {r['customer']} & {r['arrive']:.0f} & "
                    f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                    f"{r['early']:.0f} & {r['late']:.0f} & "
                    f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
        tex += f"\\multicolumn{{2}}{{r}}{{车 {v['vehicle']} 小计}} & travel={v['travel']:.0f} & demand={v['demand']:.0f}/{CAPACITY} & & pen={v['penalty']:.0f} & \\\\\n"
        tex += "\\midrule\n"
    tex += (f"\\multicolumn{{6}}{{r}}{{总 travel}} & {best_res['travel']:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{总 penalty}} & {best_res['penalty']:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{目标 J}} & {best_res['J']:.0f} & \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_04_q4_routes.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_04_q4_routes.csv + .tex", flush=True)

    # ---- K 灵敏度表 ----
    df_sens = pd.DataFrame([
        dict(K=k, J=v["J"], travel=v["travel"], penalty=v["penalty"],
             n_routes=sum(1 for r in v["routes"] if r))
        for k, v in K_results.items()
    ])
    df_sens.to_csv(OUT_TABLE / "tab_04_q4_K_sensitivity.csv", index=False, encoding="utf-8-sig")
    tex2 = ("\\begin{table}[htbp]\n\\centering\n"
            "\\caption{问题 4 · 车辆数 K 灵敏度分析}\\label{tab:q4_K_sens}\n"
            "\\begin{tabular}{ccccc}\n\\toprule\n"
            "K & 路线数 & travel & penalty & J = travel + penalty \\\\\n\\midrule\n")
    for _, row in df_sens.iterrows():
        tex2 += f"{int(row['K'])} & {int(row['n_routes'])} & {row['travel']:.0f} & {row['penalty']:.0f} & {row['J']:.0f} \\\\\n"
    tex2 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT_TABLE / "tab_04_q4_K_sensitivity.tex").write_text(tex2, encoding="utf-8")
    print(f"[写出] tables/tab_04_q4_K_sensitivity.csv + .tex", flush=True)

    # ---- 图 1：多车路径图 ----
    fig, ax = plt.subplots(figsize=(11, 11))
    K_total = len(best_res["routes"])
    cmap = plt.colormaps.get_cmap("tab10")
    # 简化：节点按 ID 分布在圆周（无地理坐标）
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {0: (0.0, 0.0)}
    for i in range(1, N + 1):
        pos[i] = (np.cos(angles[i - 1]) * 1.3, np.sin(angles[i - 1]) * 1.3)
    # 画 depot
    ax.scatter(0, 0, s=400, c="#444", marker="s", edgecolor="black", zorder=5,
               label="depot")
    ax.text(0, 0, "0", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    # 画每辆车
    for k_idx, r in enumerate(best_res["routes"]):
        if not r: continue
        color = cmap(k_idx % 10)
        full = [0] + r + [0]
        for a, b in zip(full[:-1], full[1:]):
            xa, ya = pos[a]; xb, yb = pos[b]
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.85))
        # 标 vehicle 用第一个客户点旁边
        first_c = r[0]
        x, y = pos[first_c]
        ax.text(x * 1.1, y * 1.1, f"V{k_idx+1}", color=color, fontsize=11, fontweight="bold")
    # 画客户节点（按是否违反着色）
    cust_pen = {}
    for v in schedule_per_vehicle:
        for s_row in v["schedule"]:
            cust_pen[s_row["customer"]] = s_row["penalty"]
    for i in range(1, N + 1):
        x, y = pos[i]
        pen_i = cust_pen.get(i, 0)
        color = "#d9534f" if pen_i > 0 else "#5cb85c"
        ax.scatter(x, y, s=200, c=color, edgecolor="black", linewidth=0.5, zorder=4)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Q4 · 最优 K^*={best_K} 多车辆路径\n"
                 f"travel={best_res['travel']:.0f}, penalty={best_res['penalty']:.0f}, "
                 f"J={best_res['J']:.0f}, 违反 {n_violators_total}/{N}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_routes.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_routes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_routes.png + .pdf", flush=True)

    # ---- 图 2：多车甘特 ----
    fig, ax = plt.subplots(figsize=(13, 0.6 * sum(len(v["schedule"]) for v in schedule_per_vehicle) + 2))
    y_offset = 0
    yticks = []; ylabels = []
    for v in schedule_per_vehicle:
        color = cmap((v["vehicle"] - 1) % 10)
        for r in v["schedule"]:
            y = y_offset
            ax.barh(y, r["tw_b"] - r["tw_a"], left=r["tw_a"], height=0.6,
                    color="#cfe5ff", edgecolor="#3a78c2", linewidth=0.4)
            sc = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else color
            ax.barh(y, r["service"], left=r["arrive"], height=0.4, color=sc, alpha=0.95)
            ax.plot([r["arrive"], r["arrive"]], [y - 0.4, y + 0.4], color="black", lw=0.7)
            yticks.append(y); ylabels.append(f"V{v['vehicle']} | C{r['customer']}")
            y_offset += 1
        y_offset += 0.5
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("时间")
    ax.set_title(f"Q4 · K^*={best_K} 多车辆甘特（绿/彩=未违反 红=违反）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_gantt.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_gantt.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_gantt.png + .pdf", flush=True)

    # ---- 图 3：K 灵敏度 ----
    Ks = sorted(K_results.keys())
    Js = [K_results[k]["J"] for k in Ks]
    travels = [K_results[k]["travel"] for k in Ks]
    penalties = [K_results[k]["penalty"] for k in Ks]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax2 = ax.twinx()
    l1, = ax.plot(Ks, Js, "o-", color="#3a78c2", lw=2, ms=10, label="J = travel + penalty")
    l2, = ax2.plot(Ks, travels, "s--", color="#5cb85c", lw=1.5, ms=7, label="travel")
    l3, = ax2.plot(Ks, penalties, "^:", color="#d9534f", lw=1.5, ms=7, label="penalty")
    for x_, y_ in zip(Ks, Js):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold", color="#3a78c2")
    # 标 K_min 与 K*
    ax.axvline(K_MIN, color="gray", lw=1, ls="--", alpha=0.6)
    ax.text(K_MIN, max(Js) * 0.95, f"K_min={K_MIN}\n(容量下界)", ha="left", fontsize=9, color="gray")
    ax.axvline(best_K, color="#d9534f", lw=1.2, ls="-", alpha=0.6)
    ax.text(best_K, max(Js) * 0.85, f"K^*={best_K}\n(J 最小)", ha="left", fontsize=9, color="#d9534f")
    ax.set_xlabel("可用车辆数 K")
    ax.set_ylabel("目标 J", color="#3a78c2")
    ax2.set_ylabel("travel / penalty 分量", color="#666")
    ax.set_xticks(Ks)
    ax.set_title(f"Q4 · 车辆数 K 灵敏度（题目硬性要求图）\n"
                 f"K^*={best_K}，最优 J={K_results[best_K]['J']:.0f}")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2, l3], loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_K_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_K_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_K_sensitivity.png + .pdf", flush=True)

    # ---- 图 4：违反分布 ----
    all_rows = []
    for v in schedule_per_vehicle:
        for r in v["schedule"]:
            all_rows.append(dict(vehicle=v["vehicle"], **r))
    cust_ids = [r["customer"] for r in all_rows]
    early_vals = [r["early"] for r in all_rows]
    late_vals = [r["late"] for r in all_rows]
    pen_vals = [r["penalty"] for r in all_rows]
    veh_ids = [r["vehicle"] for r in all_rows]
    x_pos = np.arange(len(cust_ids))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x_pos, late_vals, color="#d9534f", label="晚到", alpha=0.85)
    ax.bar(x_pos, [-e for e in early_vals], color="#f0ad4e", label="早到", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"V{v}|C{c}" for v, c in zip(veh_ids, cust_ids)], fontsize=6, rotation=60)
    ax.set_xlabel("车辆 | 客户（按访问顺序）")
    ax.set_ylabel("违反量（早到取负 / 晚到取正）")
    ax.set_title(f"Q4 · 50 客户时间窗违反分布（K^*={best_K}）\n"
                 f"违反客户 {n_violators_total}/{N}, 总惩罚 = {best_res['penalty']:.0f}")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    top5 = sorted(range(len(pen_vals)), key=lambda i: -pen_vals[i])[:5]
    for i in top5:
        if pen_vals[i] > 0:
            label_y = late_vals[i] if late_vals[i] > 0 else -early_vals[i]
            ax.annotate(f"罚{pen_vals[i]:.0f}", (x_pos[i], label_y),
                        textcoords="offset points",
                        xytext=(0, 6 if label_y > 0 else -10),
                        ha="center", fontsize=8, color="#444", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_violation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_violation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_violation.png + .pdf", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
