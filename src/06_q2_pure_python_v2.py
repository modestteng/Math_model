"""
问题 2 v2：含时间窗惩罚的单车辆调度（纯 Python · 强化求解器）

相对 v1 (`05_q2_pure_python.py`) 的改进
----
  1) 32 个 warm start（v1 仅 13 个）：随机 + NN(距离) + NN(时间窗紧迫) + a_i/b_i/(a+b)/2 排序 + 罚意识贪心
  2) **LNS（大邻域搜索 / ruin-and-recreate）**：每轮摧毁 3-5 个"高罚"客户，用 regret-2 贪心重插
     —— 这是相比 v1 的核心改动；单点 swap/2-opt 在二次罚地形里容易陷局部最优
  3) **多 seed 长退火 SA**：8 seeds，T0=200, α=0.998, iter/T=600（v1 仅 1 seed, T0=80, α=0.995, iter/T=400）
  4) **3-opt 子段反向**：邻域增加；与原 swap/insert/reverse 等概率
  5) **perturbation kick**：每隔若干轮做随机 3-shuffle 跳出局部最优
  6) 全程基于 J = travel + tw_penalty 评估

输出
----
  results/改进模型/q2_pure_python_v2.json
  tables/tab_02b_q2_v2_schedule.csv (+ .tex)
  figures/fig_02b_q2_v2_route.png  (+ .pdf)
  figures/fig_02b_q2_v2_compare.png (+ .pdf)   v1 vs v2 J 对比柱状图
  figures/fig_02b_q2_v2_convergence.png (+ .pdf)
  figures/fig_02b_q2_v2_lns_history.png (+ .pdf)  LNS 每轮 best J 曲线
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
OUT_RESULT = ROOT / "results/改进模型"
OUT_TABLE = ROOT / "tables"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_TABLE, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

N = 15
M1, M2 = 10.0, 20.0
SEED = 20260426

# ------------------------------------------------------------
# 1. 数据
# ------------------------------------------------------------
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_full[: N + 1, : N + 1].astype(float)
nodes = nodes_raw.iloc[: N + 1].copy()
A = nodes["tw_a"].values.astype(float)
B = nodes["tw_b"].values.astype(float)
S = nodes["service"].values.astype(float)


# ------------------------------------------------------------
# 2. 评估函数
# ------------------------------------------------------------
def evaluate(perm, with_detail: bool = False):
    travel = 0.0
    penalty = 0.0
    cur = 0.0
    last = 0
    rows = []
    per_pen = []
    for i in perm:
        tt = T[last, i]; cur += tt; travel += tt
        ai, bi = A[i], B[i]
        early = max(0.0, ai - cur); late = max(0.0, cur - bi)
        pen = M1 * early ** 2 + M2 * late ** 2
        penalty += pen
        per_pen.append((int(i), pen))
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
    """返回 [(customer, penalty), ...]，按访问顺序。"""
    pen_list = []
    cur = 0.0; last = 0
    for i in perm:
        cur += T[last, i]
        early = max(0.0, A[i] - cur); late = max(0.0, cur - B[i])
        pen = M1 * early ** 2 + M2 * late ** 2
        pen_list.append((int(i), pen))
        cur += S[i]; last = i
    return pen_list


# ------------------------------------------------------------
# 3. Warm start 集合（32 个）
# ------------------------------------------------------------
def warm_starts(rng):
    starts = []
    # 16 个随机
    for _ in range(16):
        starts.append(list(rng.permutation(N) + 1))
    # NN 距离
    unv = set(range(1, N + 1)); cur = 0; route = []
    while unv:
        nxt = min(unv, key=lambda j: T[cur, j])
        route.append(nxt); unv.discard(nxt); cur = nxt
    starts.append(route)
    # NN 综合分
    for w_late, w_early in [(0.6, 0.2), (1.0, 0.5), (2.0, 0.5), (0.3, 0.0)]:
        unv = set(range(1, N + 1)); cur = 0; t = 0; route = []
        while unv:
            def score(j):
                arr = t + T[cur, j]
                late = max(0.0, arr - B[j]); early = max(0.0, A[j] - arr)
                return T[cur, j] + w_late * late + w_early * early
            nxt = min(unv, key=score)
            route.append(nxt); t += T[cur, nxt] + S[nxt]; cur = nxt; unv.discard(nxt)
        starts.append(route)
    # 排序型
    starts.append(sorted(range(1, N + 1), key=lambda i: A[i]))
    starts.append(sorted(range(1, N + 1), key=lambda i: B[i]))
    starts.append(sorted(range(1, N + 1), key=lambda i: (A[i] + B[i]) / 2))
    starts.append(sorted(range(1, N + 1), key=lambda i: B[i] - A[i]))   # 窗宽升序
    starts.append(sorted(range(1, N + 1), key=lambda i: (A[i], B[i])))
    # 罚意识贪心：每步选"加入后预计罚增量最小"的客户
    unv = set(range(1, N + 1)); cur = 0; t = 0; route = []
    while unv:
        def regret(j):
            arr = t + T[cur, j]
            late = max(0.0, arr - B[j]); early = max(0.0, A[j] - arr)
            return T[cur, j] + 0.1 * (M1 * early ** 2 + M2 * late ** 2)
        nxt = min(unv, key=regret)
        route.append(nxt); t += T[cur, nxt] + S[nxt]; cur = nxt; unv.discard(nxt)
    starts.append(route)
    return starts


# ------------------------------------------------------------
# 4. 局部搜索算子（基于 J）
# ------------------------------------------------------------
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
                seg = best[i:i + L]
                base = best[:i] + best[i + L:]
                for j in range(0, len(base) + 1):
                    if j == i:
                        continue
                    cand = base[:j] + seg + base[j:]
                    Jc = evaluate(cand)[2]
                    if Jc < best_J - 1e-9:
                        best = cand; best_J = Jc; improved = True; iters += 1
                        break
                if improved:
                    break
            if improved:
                break
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
    """3-opt：把 perm 切 3 段，重排 4 种非平凡顺序，取最优。"""
    n = len(perm); best = list(perm); best_J = evaluate(best)[2]; iters = 0
    improved = True
    while improved:
        improved = False
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A_, B_, C_ = best[:i + 1], best[i + 1:j + 1], best[j + 1:k + 1]
                    D_ = best[k + 1:]
                    cands = [
                        A_ + B_[::-1] + C_ + D_,
                        A_ + B_ + C_[::-1] + D_,
                        A_ + B_[::-1] + C_[::-1] + D_,
                        A_ + C_ + B_ + D_,
                        A_ + C_[::-1] + B_[::-1] + D_,
                    ]
                    for cand in cands:
                        Jc = evaluate(cand)[2]
                        if Jc < best_J - 1e-9:
                            best = cand; best_J = Jc; improved = True; iters += 1
                            break
                    if improved: break
                if improved: break
            if improved: break
    return best, best_J, iters


def polish(perm):
    """2-opt → Or-opt → swap 反复直到全部不动（轻量版，3-opt 只用在最终精修）。"""
    cur = list(perm); total = 0
    while True:
        cur, _, k1 = two_opt_J(cur)
        cur, _, k2 = or_opt_J(cur)
        cur, _, k3 = swap_J(cur)
        total += k1 + k2 + k3
        if k1 + k2 + k3 == 0:
            return cur, evaluate(cur)[2], total


def polish_full(perm):
    """重型抛光：在 polish 基础上再加 3-opt（仅最终精修用）。"""
    cur, _, _ = polish(perm)
    while True:
        cur, _, k1 = three_opt_segment(cur)
        cur, _, k2 = two_opt_J(cur)
        cur, _, k3 = or_opt_J(cur)
        cur, _, k4 = swap_J(cur)
        if k1 + k2 + k3 + k4 == 0:
            return cur, evaluate(cur)[2], 0


# ------------------------------------------------------------
# 5. LNS：ruin-and-recreate
# ------------------------------------------------------------
def regret2_insert(partial, removed):
    """把 removed 列表里的客户按 regret-2 (best - 2nd-best) 顺序插入 partial。"""
    cur = list(partial)
    rem = list(removed)
    while rem:
        # 对每个待插客户算其最佳/次佳插入位置
        best_choice = None  # (regret, customer, best_pos, best_J)
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
            score = -regret + best_J * 1e-3   # 主要看 regret 大者优先
            if best_choice is None or score < best_choice[0]:
                best_choice = (score, c, costs[0][1], best_J)
        _, c, pos, _ = best_choice
        cur = cur[:pos] + [c] + cur[pos:]
        rem.remove(c)
    return cur


def lns(perm, rng, n_iter=300, ruin_min=3, ruin_max=5,
        accept_worse_T=20.0, T_decay=0.995):
    """Ruin-and-recreate 大邻域搜索。
    每轮：
      - 选 k 个"高罚"客户（一半概率）或随机（另一半），从 perm 移除
      - regret-2 重插
      - polish
      - 接受准则：J 下降必接受；上升以 exp(-ΔJ / T) 概率接受（小幅多样化）
    """
    cur = list(perm); cur_J = evaluate(cur)[2]
    best = list(cur); best_J = cur_J
    history = [best_J]
    T_now = accept_worse_T
    for it in range(n_iter):
        k = int(rng.integers(ruin_min, ruin_max + 1))
        if rng.random() < 0.5:
            # 按罚高低优先摧毁
            pen_list = per_customer_penalty(cur)
            pen_list.sort(key=lambda x: -x[1])
            top_pool = [c for c, _ in pen_list[:max(k * 2, 6)]]
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
        # perturbation kick：每 50 轮随机 shuffle 3 个位置
        if (it + 1) % 50 == 0 and rng.random() < 0.5:
            idxs = rng.choice(len(cur), size=3, replace=False)
            vals = [cur[i] for i in idxs]
            rng.shuffle(vals)
            kicked = list(cur)
            for i, v in zip(idxs, vals):
                kicked[i] = v
            kicked, kicked_J, _ = polish(kicked)
            if kicked_J < cur_J:
                cur = kicked; cur_J = kicked_J
                if cur_J < best_J:
                    best = list(cur); best_J = cur_J
    return best, best_J, history


# ------------------------------------------------------------
# 6. SA on permutation（更长 + 多种邻域）
# ------------------------------------------------------------
def sa_perm(perm0, rng, T0=200.0, Tf=1e-3, alpha=0.998, n_iter_per_T=600):
    cur = list(perm0); cur_J = evaluate(cur)[2]
    best = list(cur); best_J = cur_J
    history = [best_J]
    T_now = T0
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
            else:  # 3opt-seg：随机切 3 段重排一种
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
    return best, best_J, history


# ------------------------------------------------------------
# 7. 主流程
# ------------------------------------------------------------
def main():
    print(f"== 问题 2 v2 ==  n={N}  M1={M1}  M2={M2}")
    rng = np.random.default_rng(SEED)

    # 7.1 多起点 polish（含 3-opt）
    starts = warm_starts(rng)
    print(f"[Warm starts] {len(starts)} 个")
    t0 = time.time()
    polished = []
    for s in starts:
        p, J, _ = polish(s)
        polished.append((J, p))
    polished.sort(key=lambda kv: kv[0])
    best_polish_J, best_polish = polished[0]
    t_polish = time.time() - t0
    print(f"[Multi-start polish (含 3-opt)] {t_polish:.2f}s  best J={best_polish_J:.2f}")
    print(f"  TOP 5 J: {[round(kv[0], 2) for kv in polished[:5]]}")

    # 7.2 LNS：以 polish top-1 起步
    t0 = time.time()
    lns_best, lns_J, lns_hist = lns(best_polish, rng,
                                    n_iter=150, ruin_min=3, ruin_max=6)
    lns_polished, lns_polished_J, _ = polish(lns_best)
    t_lns = time.time() - t0
    print(f"[LNS + polish] {t_lns:.2f}s  J(LNS)={lns_J:.2f}  J(LNS+polish)={lns_polished_J:.2f}")

    # 7.3 多 seed 长退火 SA：3 seeds
    t0 = time.time()
    seeds = list(range(SEED, SEED + 3))
    sa_pool = []
    sa_history_best = None
    sa_best_J_overall = float("inf")
    for sd in seeds:
        rng_s = np.random.default_rng(sd)
        # 起点：在 polish 解 + LNS 解 中随机挑
        seed_perm = lns_polished if rng_s.random() < 0.5 else best_polish
        sa_b, sa_J, sa_hist = sa_perm(seed_perm, rng_s,
                                       T0=200.0, Tf=1e-3,
                                       alpha=0.995, n_iter_per_T=300)
        sa_b_p, sa_J_p, _ = polish(sa_b)
        sa_pool.append((sa_J_p, sa_b_p, sd))
        if sa_J_p < sa_best_J_overall:
            sa_best_J_overall = sa_J_p
            sa_history_best = sa_hist
        print(f"  seed={sd}  SA={sa_J:.2f}  SA+polish={sa_J_p:.2f}")
    sa_pool.sort(key=lambda kv: kv[0])
    t_sa = time.time() - t0
    print(f"[SA × 8 seeds + polish] {t_sa:.2f}s  best J={sa_pool[0][0]:.2f}")

    # 7.4 最终精修：在所有候选最优解上跑一次重型 polish（含 3-opt）
    candidates = [
        ("Multi-start polish", best_polish_J, best_polish),
        ("LNS + polish", lns_polished_J, lns_polished),
        (f"SA seed={sa_pool[0][2]} + polish", sa_pool[0][0], sa_pool[0][1]),
    ]
    candidates.sort(key=lambda x: x[1])
    source, final_J, final_perm = candidates[0]
    # 最终再上 3-opt
    print(f"[Final 3-opt 精修] 起点 J={final_J:.2f}", flush=True)
    final_perm, final_J, _ = polish_full(final_perm)
    print(f"[Final 3-opt 精修] 终点 J={final_J:.2f}", flush=True)
    source = source + " + 3-opt 精修"

    travel, penalty, J, schedule = evaluate(final_perm, with_detail=True)
    full_route = [0] + list(final_perm) + [0]

    # v1 结果（用于对比）
    V1_J = 84121.0
    V1_TRAVEL = 31.0
    V1_PEN = 84090.0
    V1_ROUTE = [0, 2, 13, 6, 5, 8, 7, 11, 10, 1, 9, 3, 12, 4, 15, 14, 0]

    print(f"\n========== 问题 2 v2 最终结果 ==========")
    print(f"  来源：{source}")
    print(f"  路径：{' -> '.join(map(str, full_route))}")
    print(f"  总运输时间 = {travel:.0f}")
    print(f"  时间窗惩罚 = {penalty:.2f}")
    print(f"  目标 J     = {J:.2f}")
    print(f"  v1 → v2 改进：J {V1_J:.0f} → {J:.0f}  (Δ={V1_J - J:+.0f}, {(V1_J - J) / V1_J * 100:+.2f}%)")
    print(f"  违反客户：")
    for r in schedule:
        if r["early"] > 0 or r["late"] > 0:
            print(f"    客户 {r['customer']:>2}  到达={r['arrive']:>4.0f}  "
                  f"窗=[{r['tw_a']:.0f},{r['tw_b']:.0f}]  "
                  f"早={r['early']:.0f} 晚={r['late']:.0f}  罚={r['penalty']:.0f}")

    # ---- 落盘 ----
    result = dict(
        problem="Q2 v2: 单车辆 + 时间窗惩罚（无容量），纯 Python · 强化求解器",
        method=source,
        n_customers=N, M1=M1, M2=M2,
        seeds=[SEED] + seeds,
        time_sec=dict(polish=round(t_polish, 3),
                      lns=round(t_lns, 3),
                      sa=round(t_sa, 3),
                      total=round(t_polish + t_lns + t_sa, 3)),
        final=dict(route=full_route,
                   perm=[int(x) for x in final_perm],
                   total_travel_time=float(travel),
                   total_tw_penalty=float(penalty),
                   objective_J=float(J)),
        schedule=schedule,
        topk_polish_J=[round(kv[0], 2) for kv in polished[:5]],
        sa_pool=[dict(seed=int(sd), J=round(jv, 2)) for jv, _, sd in sa_pool],
        v1_baseline=dict(J=V1_J, travel=V1_TRAVEL, penalty=V1_PEN, route=V1_ROUTE),
        improvement_vs_v1=dict(
            delta_J=float(V1_J - J),
            relative_pct=round((V1_J - J) / V1_J * 100, 4),
        ),
    )
    out_json = OUT_RESULT / "q2_pure_python_v2.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}")

    # ---- 表 ----
    df = pd.DataFrame(schedule)
    df.to_csv(OUT_TABLE / "tab_02b_q2_v2_schedule.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{问题 2 v2 单车辆调度（含时间窗违反）}\\label{tab:q2_v2_schedule}\n"
           "\\begin{tabular}{ccccccc}\n\\toprule\n"
           "客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for r in schedule:
        tex += (f"{r['customer']} & {r['arrive']:.0f} & "
                f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                f"{r['early']:.0f} & {r['late']:.0f} & "
                f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
    tex += ("\\midrule\n"
            f"\\multicolumn{{6}}{{r}}{{总运输时间}} & {travel:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{总惩罚}} & {penalty:.0f} \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{目标 J}} & {J:.0f} \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_02b_q2_v2_schedule.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_02b_q2_v2_schedule.csv + .tex")

    # ---- 图 1：路径 + 甘特 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    ax = axes[0]
    K = len(full_route) - 1
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    xs, ys = np.cos(angles), np.sin(angles)
    for k in range(K):
        nxt = (k + 1) % K
        ax.annotate("", xy=(xs[nxt], ys[nxt]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", color="#3a78c2", lw=1.4))
    for k in range(K):
        nid = full_route[k]
        color = "#888"
        if nid != 0:
            r = next(rr for rr in schedule if rr["customer"] == nid)
            color = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else "#f0ad4e"
        ax.scatter(xs[k], ys[k], s=380, c=color, edgecolor="black", zorder=3)
        ax.text(xs[k], ys[k], str(nid), ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"问题 2 v2 路径（红=违反 橙=正常 灰=depot）\n"
                 f"travel={travel:.0f}, penalty={penalty:.0f}, J={J:.0f}")

    ax = axes[1]
    yticks, ylabels = [], []
    for k, r in enumerate(schedule):
        y = k
        ax.barh(y, r["tw_b"] - r["tw_a"], left=r["tw_a"], height=0.5,
                color="#cfe5ff", edgecolor="#3a78c2", linewidth=0.8)
        sc = "#d9534f" if (r["early"] > 0 or r["late"] > 0) else "#5cb85c"
        ax.barh(y, r["service"], left=r["arrive"], height=0.32, color=sc, alpha=0.95)
        ax.plot([r["arrive"], r["arrive"]], [y - 0.32, y + 0.32], color="black", lw=1.2)
        yticks.append(y); ylabels.append(f"客户 {r['customer']}")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.invert_yaxis(); ax.set_xlabel("时间")
    ax.set_title("时间窗 vs 实际到达（绿=未违反 红=违反）")
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("问题 2 v2：含时间窗惩罚的单车辆调度（多起点+LNS+多 seed SA）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_route.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_route.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 2：v1 vs v2 对比柱状图 ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["v1 (Multi-start + SA)", "v2 (Multi-start + LNS + SA×8)"]
    Js = [V1_J, J]
    pens = [V1_PEN, penalty]
    travels = [V1_TRAVEL, travel]
    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width / 2, Js, width, label="目标 J", color="#3a78c2")
    ax.bar(x + width / 2, pens, width, label="时间窗惩罚", color="#d9534f", alpha=0.7)
    for xi, jv in zip(x - width / 2, Js):
        ax.text(xi, jv, f"{jv:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for xi, pv, tv in zip(x + width / 2, pens, travels):
        ax.text(xi, pv, f"{pv:.0f}\n(travel={tv:.0f})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("数值")
    ax.set_title(f"v1 → v2 改进对比（ΔJ = {V1_J - J:+.0f}, {(V1_J - J) / V1_J * 100:+.2f}%）")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_compare.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 3：SA 收敛 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sa_history_best, color="#5cb85c", lw=1.4)
    ax.set_xlabel("外层降温步")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 2 v2 · 最优 seed 的 SA 收敛曲线（α=0.998, iter/T=600）\n"
                 f"final J = {sa_best_J_overall:.0f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 图 4：LNS 历史 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lns_hist, color="#8e44ad", lw=1.2)
    ax.set_xlabel("LNS 迭代轮数")
    ax.set_ylabel("最佳 J")
    ax.set_title(f"问题 2 v2 · LNS 大邻域搜索历史曲线\n"
                 f"start J = {lns_hist[0]:.0f},  end J = {lns_hist[-1]:.0f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_lns_history.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_02b_q2_v2_lns_history.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[写出] figures/fig_02b_q2_v2_route.png + .pdf")
    print(f"[写出] figures/fig_02b_q2_v2_compare.png + .pdf")
    print(f"[写出] figures/fig_02b_q2_v2_convergence.png + .pdf")
    print(f"[写出] figures/fig_02b_q2_v2_lns_history.png + .pdf")


if __name__ == "__main__":
    main()
