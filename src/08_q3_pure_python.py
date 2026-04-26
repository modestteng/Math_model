"""
问题 3 · 第 1 步：纯 Python 求解 n=50 单车辆 + 时间窗（基线）

设计
----
  把 Q2 v2（src/06_q2_pure_python_v2.py）框架放大到 n=50：
    多起点 polish (40 个起点) + LNS (500 轮 ruin-and-recreate) + 多 seed SA + 3-opt 精修
  本步**仅**给出经典启发式基线 J*，作为下一步 Kaiwu SDK / CIM 大规模分解的对照。

  量子比特说明（铁律 §二.5）：
    n=50 直接 one-hot QUBO 需 n²=2500 比特，远超 CIM CPQC-550 真机 550 比特上限，
    故必须分解（下一步 plan）；本步纯 Python，不涉及 QUBO，比特数为 0。

输出
----
  results/基础模型/q3_pure_python.json
  tables/tab_03_q3_schedule.csv (+ .tex)
  figures/fig_03_q3_route.png             (+ .pdf)  路径 + 时间窗甘特
  figures/fig_03_q3_convergence.png       (+ .pdf)  SA 收敛
  figures/fig_03_q3_lns_history.png       (+ .pdf)  LNS 历史
  figures/fig_03_q3_violation_distribution.png (+ .pdf)  50 客户违反柱状图
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

# ---- 参数 ----
N = 50
M1, M2 = 10.0, 20.0
SEED = 20260426
WARM_STARTS_RANDOM = 32
LNS_ITERS = 500
LNS_RUIN_MIN, LNS_RUIN_MAX = 5, 10
SA_SEEDS = 3
SA_T0, SA_ALPHA, SA_ITER_PER_T = 300.0, 0.997, 400

# ---- 数据 ----
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_full[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)


def evaluate(perm, with_detail: bool = False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []
    for i in perm:
        tt = T[last, i]; cur += tt; travel += tt
        ai, bi = A[i], B[i]
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen
        if with_detail:
            rows.append(dict(customer=int(i), arrive=float(cur),
                             tw_a=float(ai), tw_b=float(bi),
                             early=float(early), late=float(late),
                             penalty=float(pen), service=float(S[i]),
                             depart=float(cur + S[i])))
        cur += S[i]; last = i
    travel += T[last, 0]
    J = travel + penalty
    if with_detail:
        return travel, penalty, J, rows
    return travel, penalty, J


def per_customer_penalty(perm):
    pen_list = []; cur = 0.0; last = 0
    for i in perm:
        cur += T[last, i]
        early = max(0.0, A[i] - cur); late = max(0.0, cur - B[i])
        pen = M1 * early ** 2 + M2 * late ** 2
        pen_list.append((int(i), pen))
        cur += S[i]; last = i
    return pen_list


def warm_starts(rng):
    starts = []
    for _ in range(WARM_STARTS_RANDOM):
        starts.append(list(rng.permutation(N) + 1))
    # NN 距离
    unv = set(range(1, N + 1)); cur = 0; route = []
    while unv:
        nxt = min(unv, key=lambda j: T[cur, j])
        route.append(nxt); unv.discard(nxt); cur = nxt
    starts.append(route)
    # NN 综合分
    for w_late, w_early in [(0.6, 0.2), (1.0, 0.5), (2.0, 0.5), (0.3, 0.0), (1.5, 0.3)]:
        unv = set(range(1, N + 1)); cur = 0; t = 0; route = []
        while unv:
            def score(j):
                arr = t + T[cur, j]
                late = max(0.0, arr - B[j]); early_ = max(0.0, A[j] - arr)
                return T[cur, j] + w_late * late + w_early * early_
            nxt = min(unv, key=score)
            route.append(nxt); t += T[cur, nxt] + S[nxt]; cur = nxt; unv.discard(nxt)
        starts.append(route)
    # 排序型
    starts.append(sorted(range(1, N + 1), key=lambda i: A[i]))
    starts.append(sorted(range(1, N + 1), key=lambda i: B[i]))
    starts.append(sorted(range(1, N + 1), key=lambda i: (A[i] + B[i]) / 2))
    starts.append(sorted(range(1, N + 1), key=lambda i: B[i] - A[i]))
    # 罚意识贪心
    unv = set(range(1, N + 1)); cur = 0; t = 0; route = []
    while unv:
        def regret(j):
            arr = t + T[cur, j]
            late = max(0.0, arr - B[j]); early_ = max(0.0, A[j] - arr)
            return T[cur, j] + 0.1 * (M1 * early_ ** 2 + M2 * late ** 2)
        nxt = min(unv, key=regret)
        route.append(nxt); t += T[cur, nxt] + S[nxt]; cur = nxt; unv.discard(nxt)
    starts.append(route)
    return starts


def two_opt_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]; iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 1, n):
                cand = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True; iters += 1
    return best, best_J, iters


def or_opt_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]; iters = 0
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
                        best = cand; best_J = Jc; improved = True; iters += 1; break
                if improved: break
            if improved: break
    return best, best_J, iters


def swap_J(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]; iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best.copy(); cand[i], cand[j] = cand[j], cand[i]
                Jc = evaluate(cand)[2]
                if Jc < best_J - 1e-9:
                    best = cand; best_J = Jc; improved = True; iters += 1
    return best, best_J, iters


def three_opt_segment(perm):
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]; iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A_, B_, C_ = best[:i + 1], best[i + 1:j + 1], best[j + 1:k + 1]
                    D_ = best[k + 1:]
                    cands = [A_ + B_[::-1] + C_ + D_,
                             A_ + B_ + C_[::-1] + D_,
                             A_ + B_[::-1] + C_[::-1] + D_,
                             A_ + C_ + B_ + D_,
                             A_ + C_[::-1] + B_[::-1] + D_]
                    for cand in cands:
                        Jc = evaluate(cand)[2]
                        if Jc < best_J - 1e-9:
                            best = cand; best_J = Jc; improved = True; iters += 1; break
                    if improved: break
                if improved: break
            if improved: break
    return best, best_J, iters


def polish(perm):
    cur = list(perm); total = 0
    while True:
        cur, _, k1 = two_opt_J(cur)
        cur, _, k2 = or_opt_J(cur)
        cur, _, k3 = swap_J(cur)
        total += k1 + k2 + k3
        if k1 + k2 + k3 == 0:
            return cur, evaluate(cur)[2], total


def polish_full(perm):
    cur, _, _ = polish(perm)
    while True:
        cur, _, k1 = three_opt_segment(cur)
        cur, _, k2 = two_opt_J(cur)
        cur, _, k3 = or_opt_J(cur)
        cur, _, k4 = swap_J(cur)
        if k1 + k2 + k3 + k4 == 0:
            return cur, evaluate(cur)[2], 0


def regret2_insert(partial, removed):
    cur = list(partial); rem = list(removed)
    while rem:
        best_choice = None
        for c in rem:
            costs = []
            for pos in range(len(cur) + 1):
                cand = cur[:pos] + [c] + cur[pos:]
                Jc = evaluate(cand)[2]
                costs.append((Jc, pos))
            costs.sort()
            best_J = costs[0][0]
            second = costs[1][0] if len(costs) > 1 else best_J
            regret = second - best_J
            score = -regret + best_J * 1e-3
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, c, costs[0][1], best_J)
        _, c, pos, _ = best_choice
        cur = cur[:pos] + [c] + cur[pos:]
        rem.remove(c)
    return cur


def lns(perm, rng, n_iter=LNS_ITERS, ruin_min=LNS_RUIN_MIN, ruin_max=LNS_RUIN_MAX,
        accept_worse_T=20.0, T_decay=0.997, log_every=25):
    cur = list(perm); cur_J = evaluate(cur)[2]
    best = list(cur); best_J = cur_J
    history = [best_J]
    T_now = accept_worse_T
    t0 = time.time()
    for it in range(n_iter):
        k = int(rng.integers(ruin_min, ruin_max + 1))
        if rng.random() < 0.5:
            pen_list = per_customer_penalty(cur)
            pen_list.sort(key=lambda x: -x[1])
            top_pool = [c for c, _ in pen_list[:max(k * 2, 10)]]
            removed = list(rng.choice(top_pool, size=min(k, len(top_pool)), replace=False))
        else:
            removed = list(rng.choice(cur, size=k, replace=False))
        partial = [x for x in cur if x not in removed]
        cand = regret2_insert(partial, removed)
        cand, cand_J, _ = polish(cand)
        dJ = cand_J - cur_J
        if dJ < 0 or rng.random() < np.exp(-dJ / max(T_now, 1e-6)):
            cur = cand; cur_J = cand_J
            if cur_J < best_J:
                best = list(cur); best_J = cur_J
        history.append(best_J)
        T_now *= T_decay
        if (it + 1) % 100 == 0 and rng.random() < 0.5:
            idxs = rng.choice(len(cur), size=5, replace=False)
            vals = [cur[i] for i in idxs]
            rng.shuffle(vals)
            kicked = list(cur)
            for i, v in zip(idxs, vals): kicked[i] = v
            kicked, kicked_J, _ = polish(kicked)
            if kicked_J < cur_J:
                cur = kicked; cur_J = kicked_J
                if cur_J < best_J:
                    best = list(cur); best_J = cur_J
        if (it + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"    LNS iter={it+1:4d}/{n_iter}  best_J={best_J:.0f}  elapsed={elapsed:.1f}s", flush=True)
    return best, best_J, history


def sa_perm(perm0, rng, T0=SA_T0, Tf=1e-3, alpha=SA_ALPHA, n_iter_per_T=SA_ITER_PER_T):
    cur = list(perm0); cur_J = evaluate(cur)[2]
    best = list(cur); best_J = cur_J
    history = [best_J]
    T_now = T0
    t0 = time.time()
    step = 0
    while T_now > Tf:
        for _ in range(n_iter_per_T):
            move = rng.choice(["swap", "insert", "reverse", "3opt-seg"])
            cand = cur.copy(); n = len(cand)
            if move == "swap":
                i, j = rng.integers(0, n, size=2)
                if i == j: continue
                cand[i], cand[j] = cand[j], cand[i]
            elif move == "insert":
                L = int(rng.integers(1, 4))
                if L >= n: continue
                i = int(rng.integers(0, n - L + 1))
                seg = cand[i:i + L]; base = cand[:i] + cand[i + L:]
                j = int(rng.integers(0, len(base) + 1))
                if j == i: continue
                cand = base[:j] + seg + base[j:]
            elif move == "reverse":
                i, j = sorted(rng.integers(0, n, size=2).tolist())
                if i == j: continue
                cand = cand[:i] + cand[i:j + 1][::-1] + cand[j + 1:]
            else:
                if n < 4: continue
                i, j, k = sorted(rng.choice(n - 1, size=3, replace=False).tolist())
                A_, B_, C_, D_ = cand[:i + 1], cand[i + 1:j + 1], cand[j + 1:k + 1], cand[k + 1:]
                cand = A_ + C_ + B_ + D_
            Jc = evaluate(cand)[2]
            dJ = Jc - cur_J
            if dJ < 0 or rng.random() < np.exp(-dJ / T_now):
                cur = cand; cur_J = Jc
                if cur_J < best_J:
                    best = list(cur); best_J = cur_J
        history.append(best_J)
        T_now *= alpha
        step += 1
        if step % 50 == 0:
            print(f"    SA step={step:4d}  T={T_now:.4f}  best_J={best_J:.0f}  elapsed={time.time() - t0:.1f}s", flush=True)
    return best, best_J, history


def main():
    print(f"== 问题 3 · 纯 Python n=50 ==  M1={M1} M2={M2}", flush=True)
    rng = np.random.default_rng(SEED)

    # 7.1 多起点 polish
    starts = warm_starts(rng)
    print(f"[Warm starts] {len(starts)} 个", flush=True)
    t0 = time.time()
    polished = []
    for k, s in enumerate(starts):
        p, J, _ = polish(s)
        polished.append((J, p))
        if (k + 1) % 5 == 0:
            print(f"    polish {k+1}/{len(starts)}  cur best={min(j for j, _ in polished):.0f}", flush=True)
    polished.sort(key=lambda kv: kv[0])
    best_polish_J, best_polish = polished[0]
    t_polish = time.time() - t0
    print(f"[Multi-start polish] {t_polish:.1f}s  best J={best_polish_J:.0f}", flush=True)
    print(f"  TOP 5 J: {[round(kv[0]) for kv in polished[:5]]}", flush=True)

    # 7.2 LNS
    print(f"\n[LNS] {LNS_ITERS} 轮，ruin {LNS_RUIN_MIN}-{LNS_RUIN_MAX}", flush=True)
    t0 = time.time()
    lns_best, lns_J, lns_hist = lns(best_polish, rng)
    lns_polished, lns_polished_J, _ = polish(lns_best)
    t_lns = time.time() - t0
    print(f"[LNS + polish] {t_lns:.1f}s  J(LNS)={lns_J:.0f}  J(LNS+polish)={lns_polished_J:.0f}", flush=True)

    # 7.3 SA × SEEDS
    t0 = time.time()
    seeds = list(range(SEED, SEED + SA_SEEDS))
    sa_pool = []
    sa_history_best = None
    sa_best_J_overall = float("inf")
    for sd in seeds:
        print(f"\n[SA seed={sd}]", flush=True)
        rng_s = np.random.default_rng(sd)
        seed_perm = lns_polished if rng_s.random() < 0.5 else best_polish
        sa_b, sa_J_, sa_hist = sa_perm(seed_perm, rng_s)
        sa_b_p, sa_J_p, _ = polish(sa_b)
        sa_pool.append((sa_J_p, sa_b_p, sd))
        if sa_J_p < sa_best_J_overall:
            sa_best_J_overall = sa_J_p
            sa_history_best = sa_hist
        print(f"  seed={sd}  SA={sa_J_:.0f}  SA+polish={sa_J_p:.0f}", flush=True)
    sa_pool.sort(key=lambda kv: kv[0])
    t_sa = time.time() - t0
    print(f"\n[SA × {SA_SEEDS} seeds + polish] {t_sa:.1f}s  best J={sa_pool[0][0]:.0f}", flush=True)

    # 7.4 选 best + 最终 3-opt 精修
    candidates = [
        ("Multi-start polish", best_polish_J, best_polish),
        ("LNS + polish", lns_polished_J, lns_polished),
        (f"SA seed={sa_pool[0][2]} + polish", sa_pool[0][0], sa_pool[0][1]),
    ]
    candidates.sort(key=lambda x: x[1])
    source, final_J, final_perm = candidates[0]
    print(f"\n[Final 3-opt 精修] 起点 J={final_J:.0f}", flush=True)
    final_perm, final_J, _ = polish_full(final_perm)
    print(f"[Final 3-opt 精修] 终点 J={final_J:.0f}", flush=True)
    source = source + " + 3-opt 精修"

    travel, penalty, J, schedule = evaluate(final_perm, with_detail=True)
    full_route = [0] + list(final_perm) + [0]

    n_violators = sum(1 for r in schedule if r["early"] > 0 or r["late"] > 0)
    print(f"\n========== 问题 3 最终结果 (n=50) ==========", flush=True)
    print(f"  来源：{source}", flush=True)
    print(f"  路径：{full_route}", flush=True)
    print(f"  总运输时间 = {travel:.0f}", flush=True)
    print(f"  时间窗惩罚 = {penalty:.0f}", flush=True)
    print(f"  目标 J     = {J:.0f}", flush=True)
    print(f"  违反客户数 = {n_violators}/{N}", flush=True)
    # 列出 top10 违反
    sorted_v = sorted([r for r in schedule if r["early"] > 0 or r["late"] > 0],
                      key=lambda r: -r["penalty"])
    print(f"  违反 TOP 10（按罚值降序）:", flush=True)
    for r in sorted_v[:10]:
        print(f"    客户 {r['customer']:>2}  到达={r['arrive']:>5.0f}  "
              f"窗=[{r['tw_a']:.0f},{r['tw_b']:.0f}]  "
              f"早={r['early']:.0f} 晚={r['late']:.0f}  罚={r['penalty']:.0f}", flush=True)

    # ---- JSON ----
    result = dict(
        problem="Q3 第 1 步: 纯 Python n=50 单车辆 + 时间窗（基线）",
        method=source,
        n_customers=N, M1=M1, M2=M2,
        seeds=[SEED] + seeds,
        params=dict(
            warm_starts=len(starts), lns_iters=LNS_ITERS,
            lns_ruin_min=LNS_RUIN_MIN, lns_ruin_max=LNS_RUIN_MAX,
            sa_seeds=SA_SEEDS, sa_T0=SA_T0, sa_alpha=SA_ALPHA,
            sa_iter_per_T=SA_ITER_PER_T,
        ),
        time_sec=dict(polish=round(t_polish, 2), lns=round(t_lns, 2),
                      sa=round(t_sa, 2),
                      total=round(t_polish + t_lns + t_sa, 2)),
        final=dict(route=full_route,
                   perm=[int(x) for x in final_perm],
                   total_travel_time=float(travel),
                   total_tw_penalty=float(penalty),
                   objective_J=float(J),
                   n_violators=int(n_violators)),
        schedule=schedule,
        topk_polish_J=[round(kv[0]) for kv in polished[:5]],
        sa_pool=[dict(seed=int(sd), J=round(jv)) for jv, _, sd in sa_pool],
        note=("此为 n=50 经典启发式基线 J*；下一步将设计大规模 QUBO 分解算法"
              "（聚类/扇区/滚动窗口）调用 Kaiwu SDK + CIM 真机，把每个子 QUBO ≤23²=529 比特，"
              "与本基线对比。比特数说明：本步纯 Python，不涉及 QUBO，比特数为 0。"),
    )
    def _to_jsonable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"not jsonable: {type(obj)}")

    out_json = OUT_RESULT / "q3_pure_python.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_to_jsonable),
        encoding="utf-8",
    )
    print(f"\n[写出] {out_json.relative_to(ROOT)}", flush=True)

    # ---- 表 ----
    df = pd.DataFrame(schedule)
    df.to_csv(OUT_TABLE / "tab_03_q3_schedule.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{问题 3 单车辆 50 客户调度（纯 Python 基线）}\\label{tab:q3_schedule}\n"
           "\\small\n\\begin{tabular}{ccccccc}\n\\toprule\n"
           "客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for r in schedule:
        tex += (f"{r['customer']} & {r['arrive']:.0f} & "
                f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                f"{r['early']:.0f} & {r['late']:.0f} & "
                f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
    tex += ("\\midrule\n"
            f"\\multicolumn{{6}}{{r}}{{总运输时间}} & {travel:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{时间窗惩罚总和}} & {penalty:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{目标 J}} & {J:.0f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_03_q3_schedule.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_03_q3_schedule.csv + .tex", flush=True)

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
    ax.set_title(f"问题 3 路径（n=50 · 红=违反 橙=正常 灰=depot）\n"
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

    fig.suptitle("问题 3：n=50 单车辆 + 时间窗惩罚（纯 Python 基线）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_route.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_route.png + .pdf", flush=True)

    # ---- 图 2：SA 收敛 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sa_history_best, color="#5cb85c", lw=1.4)
    ax.set_xlabel("外层降温步")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 3 · 最优 seed 的 SA 收敛曲线\n"
                 f"T0={SA_T0}, α={SA_ALPHA}, iter/T={SA_ITER_PER_T},  final J = {sa_best_J_overall:.0f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_convergence.png + .pdf", flush=True)

    # ---- 图 3：LNS 历史 ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lns_hist, color="#8e44ad", lw=1.0)
    ax.set_xlabel("LNS 迭代轮数")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 3 · LNS 大邻域搜索历史\n"
                 f"start J = {lns_hist[0]:.0f},  end J = {lns_hist[-1]:.0f}, "
                 f"ruin={LNS_RUIN_MIN}-{LNS_RUIN_MAX}, n_iter={LNS_ITERS}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_lns_history.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_lns_history.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_lns_history.png + .pdf", flush=True)

    # ---- 图 4：违反分布柱状图 ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    customer_ids = [r["customer"] for r in schedule]
    early_vals = [r["early"] for r in schedule]
    late_vals = [r["late"] for r in schedule]
    pen_vals = [r["penalty"] for r in schedule]
    x_pos = np.arange(len(customer_ids))

    ax.bar(x_pos, late_vals, color="#d9534f", label="晚到 (单位时间)", alpha=0.85)
    ax.bar(x_pos, [-e for e in early_vals], color="#f0ad4e", label="早到 (单位时间)", alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(customer_ids, fontsize=7, rotation=45)
    ax.set_xlabel("客户编号（按访问顺序）")
    ax.set_ylabel("违反量（早到取负 / 晚到取正）")
    ax.set_title(f"问题 3 · 50 客户时间窗违反分布\n"
                 f"违反客户 {n_violators}/{N},  总惩罚 = {penalty:.0f},  最严重客户 罚={max(pen_vals):.0f}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # 在罚最大的 5 个客户上标注罚值
    top5_idx = sorted(range(len(pen_vals)), key=lambda i: -pen_vals[i])[:5]
    for i in top5_idx:
        if pen_vals[i] > 0:
            label_y = late_vals[i] if late_vals[i] > 0 else -early_vals[i]
            ax.annotate(f"罚{pen_vals[i]:.0f}",
                        (x_pos[i], label_y),
                        textcoords="offset points", xytext=(0, 6 if label_y > 0 else -10),
                        ha="center", fontsize=8, color="#444", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_03_q3_violation_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_03_q3_violation_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_03_q3_violation_distribution.png + .pdf", flush=True)

    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
