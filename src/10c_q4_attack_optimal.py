"""
Q4 第 1 步增强：以 main.py 参考解作 warm start + 扩 K=[5..10] + 长程 LNS × 多 seed
目标：突破 main.py 参考解 J=7149（K=7）。
"""
from __future__ import annotations
import json, time, importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# 复用 10_q4_pure_python.py 的所有函数与全局
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "src"))
mod = importlib.import_module("10_q4_pure_python")
T = mod.T; A = mod.A; B = mod.B; S = mod.S; D = mod.D
N = mod.N; M1 = mod.M1; M2 = mod.M2; CAPACITY = mod.CAPACITY
SEED = 20260426

OUT_RESULT = ROOT / "results/基础模型"
OUT_SENS = ROOT / "results/灵敏度分析"
OUT_TABLE = ROOT / "tables"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_SENS, OUT_TABLE, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

# main.py 参考解 K=7
REF_K7 = [
    [15, 2, 21, 40, 6, 37, 17],
    [31, 30, 9, 34, 35, 20, 32],
    [27, 16, 44, 38, 14, 43, 42, 13],
    [47, 19, 36, 49, 11, 10, 1],
    [33, 25, 39, 23, 22, 41, 4],
    [28, 29, 12, 3, 50, 26, 24],
    [5, 45, 7, 18, 8, 46, 48],
]

# 攻击参数
K_RANGE = [5, 6, 7, 8, 9, 10]
LNS_ITERS = 500
SA_SEEDS = 4
M_VEHICLE = 1000.0  # 综合目标 = M·K + travel + pen


def adjust_to_K_keep_assignment(routes, target_K, rng):
    """与 mod.adjust_to_K 同语义，但优先保持参考分配。"""
    return mod.adjust_to_K(routes, target_K, rng)


def collect_warm_starts_aug(K_target, rng):
    starts = mod.collect_warm_starts(K_target, rng)
    # 加 REF_K7（按 K 调整）
    ref_adjusted = adjust_to_K_keep_assignment([list(r) for r in REF_K7], K_target, rng)
    if mod.feasible_capacity(ref_adjusted) and len(ref_adjusted) == K_target:
        starts.append(("ref_main_py", ref_adjusted))
    # 把 REF 反向 / 按 a_i 重新排
    for variant_name, transform in [
        ("ref_reversed", lambda rs: [r[::-1] for r in rs]),
        ("ref_sortA", lambda rs: [sorted(r, key=lambda c: A[c]) for r in rs]),
    ]:
        adj = adjust_to_K_keep_assignment([list(r) for r in transform(REF_K7)], K_target, rng)
        if mod.feasible_capacity(adj) and len(adj) == K_target:
            starts.append((variant_name, adj))
    return starts


def solve_for_K_aug(K_target, rng_master):
    starts = collect_warm_starts_aug(K_target, rng_master)
    if not starts:
        print(f"  [K={K_target}] 无可行 warm start，跳过", flush=True)
        return None
    print(f"  [K={K_target}] {len(starts)} 个 warm start（含 ref_main_py 等）", flush=True)

    # 阶段 1：每个 warm start polish + cross
    candidates = []
    for name, routes in starts:
        rp = mod.polish_all_vehicles(routes)
        rp, _ = mod.cross_vehicle_optimize(rp, max_iter=8)
        rp = mod.polish_all_vehicles(rp)
        _, _, J, _ = mod.evaluate_solution(rp)
        candidates.append((J, rp, name))
    candidates.sort(key=lambda x: x[0])
    base_J, base_routes, base_name = candidates[0]
    print(f"  [K={K_target}] 多起点 polish: best J={base_J:.0f} (来源={base_name})", flush=True)
    print(f"    TOP-5 J = {[(round(c[0]), c[2]) for c in candidates[:5]]}", flush=True)

    # 阶段 2：LNS × seeds
    overall_best_J = base_J
    overall_best_routes = [list(r) for r in base_routes]
    for sd_idx in range(SA_SEEDS):
        rng_lns = np.random.default_rng(SEED + K_target * 1000 + sd_idx)
        seed_start_idx = sd_idx % min(len(candidates), 4)
        start_routes = [list(r) for r in candidates[seed_start_idx][1]]
        lns_best, lns_J, _ = mod.lns_cross_vehicle(
            start_routes, rng_lns, n_iter=LNS_ITERS, ruin_min=3, ruin_max=8
        )
        lns_best = mod.polish_all_vehicles(lns_best)
        lns_best, _ = mod.cross_vehicle_optimize(lns_best, max_iter=8)
        lns_best = mod.polish_all_vehicles(lns_best)
        _, _, lp_J, _ = mod.evaluate_solution(lns_best)
        print(f"    [K={K_target}] seed#{sd_idx} (start={candidates[seed_start_idx][2]}): J={lp_J:.0f}", flush=True)
        if lp_J < overall_best_J - 1e-9:
            overall_best_J = lp_J
            overall_best_routes = [list(r) for r in lns_best]

    # 收尾：再 LNS 一轮短程 + cross
    rng_final = np.random.default_rng(SEED + K_target * 9999)
    final_routes, final_J, _ = mod.lns_cross_vehicle(
        overall_best_routes, rng_final, n_iter=200, ruin_min=2, ruin_max=4
    )
    final_routes = mod.polish_all_vehicles(final_routes)
    final_routes, _ = mod.cross_vehicle_optimize(final_routes, max_iter=10)
    final_routes = mod.polish_all_vehicles(final_routes)
    travel, pen, J, _ = mod.evaluate_solution(final_routes)
    if J < overall_best_J - 1e-9:
        overall_best_J = J
        overall_best_routes = final_routes
    travel, pen, J, _ = mod.evaluate_solution(overall_best_routes)
    obj_with_M = M_VEHICLE * K_target + J
    print(f"  [K={K_target}] 最终 J(travel+pen)={J:.0f}, travel={travel:.0f}, pen={pen:.0f}, "
          f"obj(M=1000)={obj_with_M:.0f}", flush=True)
    return dict(K=K_target, routes=overall_best_routes,
                travel=float(travel), penalty=float(pen),
                J_inner=float(J), objective_M=float(obj_with_M))


def main():
    print(f"== Q4 第 1 步增强 ==  K_RANGE={K_RANGE}  LNS_ITERS={LNS_ITERS}  seeds={SA_SEEDS}  M_VEHICLE={M_VEHICLE}", flush=True)
    rng_master = np.random.default_rng(SEED)
    K_results = {}
    t_total = time.time()
    for K in K_RANGE:
        print(f"\n[K = {K}] ----------------------------------------", flush=True)
        res = solve_for_K_aug(K, rng_master)
        if res is None: continue
        K_results[K] = res
    t_total = time.time() - t_total
    print(f"\n[All K] 总耗时 {t_total:.1f}s", flush=True)

    # 选 K* by 综合目标 (M·K + J)
    bestK = min(K_results.keys(), key=lambda k: K_results[k]["objective_M"])
    best = K_results[bestK]
    print(f"\n========== Q4 攻击最优 K* = {bestK} ==========", flush=True)
    print(f"  travel={best['travel']:.0f}, pen={best['penalty']:.0f}, "
          f"J(travel+pen)={best['J_inner']:.0f}, obj(M=1000)={best['objective_M']:.0f}", flush=True)

    # vs main.py 参考
    REF_OBJ = {5: 16825, 6: 7568, 7: 7149, 8: 8160, 9: 9127, 10: 10126}
    REF_TR = {5: 85, 6: 98, 7: 109, 8: 130, 9: 127, 10: 126}
    REF_PE = {5: 11740, 6: 1470, 7: 40, 8: 30, 9: 0, 10: 0}
    print(f"\n  K vs main.py 参考解：", flush=True)
    print(f"  K | mine travel | mine pen | mine obj | main travel | main pen | main obj | ΔObj  | win?", flush=True)
    for K in sorted(K_results.keys()):
        r = K_results[K]; ro = REF_OBJ.get(K)
        delta = r["objective_M"] - ro if ro else None
        win = "我赢" if delta is not None and delta < 0 else ("打平" if delta == 0 else "main 赢")
        print(f"  {K} | {r['travel']:>11.0f} | {r['penalty']:>8.0f} | {r['objective_M']:>8.0f} | "
              f"{REF_TR[K]:>11} | {REF_PE[K]:>8} | {ro:>8} | {delta:>+5.0f} | {win}", flush=True)

    # 输出
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    output = dict(
        problem="Q4 第 1 步增强（攻击 main.py 参考解）",
        M_vehicle=M_VEHICLE, K_RANGE=K_RANGE,
        time_sec=round(t_total, 2),
        bestK=bestK,
        best_solution=dict(
            K=bestK,
            routes=[[0] + [int(c) for c in r] + [0] for r in best["routes"]],
            travel=best["travel"], penalty=best["penalty"],
            J_inner=best["J_inner"], obj_M=best["objective_M"],
        ),
        K_sensitivity={int(k): dict(travel=v["travel"], penalty=v["penalty"],
                                    J_inner=v["J_inner"], obj_M=v["objective_M"],
                                    routes=[[0] + [int(c) for c in r] + [0] for r in v["routes"]])
                       for k, v in K_results.items()},
        comparison_with_main_py=dict(
            REF_OBJ=REF_OBJ, REF_TR=REF_TR, REF_PE=REF_PE,
            wins={int(k): K_results[k]["objective_M"] - REF_OBJ.get(k, 0)
                  for k in K_results},
        ),
    )
    out_json = OUT_RESULT / "q4_attack_optimal.json"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}", flush=True)

    # K vs obj 图
    Ks = sorted(K_results.keys())
    mine_obj = [K_results[k]["objective_M"] for k in Ks]
    ref_obj = [REF_OBJ[k] for k in Ks]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(Ks, mine_obj, "o-", color="#3a78c2", lw=2, ms=10, label="本文（增强版）")
    ax.plot(Ks, ref_obj, "s--", color="#d9534f", lw=1.5, ms=8, label="main.py 参考解")
    for x_, y_ in zip(Ks, mine_obj):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold", color="#3a78c2")
    for x_, y_ in zip(Ks, ref_obj):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, fontweight="bold", color="#d9534f")
    ax.set_xlabel("可用车辆数 K"); ax.set_ylabel("综合目标 = 1000·K + travel + penalty")
    ax.set_xticks(Ks)
    ax.set_title(f"Q4 攻击对比：本文 vs main.py 参考解\n"
                 f"K* = {bestK}, 最优综合目标 = {best['objective_M']:.0f}")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_attack_compare.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_attack_compare.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_attack_compare.png + .pdf", flush=True)
    print("\n[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
