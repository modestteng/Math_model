"""
Q4 论文最终交付：从 q4_attack_optimal.json 取 K* = 7 J=7149 方案，
重新生成与论文一致的所有表格 + 图（覆盖 R-Q4-002 残留的 K=8 版本）。
"""
from __future__ import annotations
import json
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

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

N = 50
M1, M2 = 10.0, 20.0
CAPACITY = 60
M_VEHICLE = 1000.0

# 数据
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_full[: N + 1, : N + 1].astype(float)
A = nodes_raw["tw_a"].values[: N + 1].astype(float)
B = nodes_raw["tw_b"].values[: N + 1].astype(float)
S = nodes_raw["service"].values[: N + 1].astype(float)
D = nodes_raw["demand"].values[: N + 1].astype(float)


def evaluate_route(route_customers, with_detail=False):
    travel = 0.0; penalty = 0.0; cur = 0.0; last = 0; rows = []; dsum = 0.0
    for i in route_customers:
        i = int(i); tt = T[last, i]; cur += tt; travel += tt
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
    travel += T[last, 0]
    if with_detail:
        return float(travel), float(penalty), float(dsum), rows
    return float(travel), float(penalty), float(dsum)


def main():
    src_json = OUT_RESULT / "q4_attack_optimal.json"
    data = json.loads(src_json.read_text(encoding="utf-8"))
    best_K = data["bestK"]
    best_routes_with_depot = data["best_solution"]["routes"]  # [[0, ..., 0], ...]
    routes = [r[1:-1] for r in best_routes_with_depot]
    travel = data["best_solution"]["travel"]
    penalty = data["best_solution"]["penalty"]
    J_inner = data["best_solution"]["J_inner"]
    obj_M = data["best_solution"]["obj_M"]

    print(f"[Finalize] K*={best_K}, travel={travel:.0f}, penalty={penalty:.0f}, J_inner={J_inner:.0f}, obj_M={obj_M:.0f}")

    schedule_per_vehicle = []
    n_violators_total = 0
    for k_idx, r in enumerate(routes):
        if not r: continue
        tr, pn, dsum, rows = evaluate_route(r, with_detail=True)
        schedule_per_vehicle.append(dict(vehicle=k_idx + 1, route=[0] + r + [0],
                                          travel=tr, penalty=pn, demand=dsum,
                                          schedule=rows))
        n_violators_total += sum(1 for x in rows if x["early"] > 0 or x["late"] > 0)

    print(f"  总违反客户：{n_violators_total}/{N}")

    # ---- 重写 q4_pure_python.json（指向 K=7 最优方案） ----
    def _to_jsonable(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not jsonable: {type(o)}")

    K_sens = data["K_sensitivity"]
    final_doc = dict(
        problem="Q4 · 多车辆 VRP（容量 + 时间窗）— 论文最终交付（来自 R-Q4-003 攻击版）",
        n_customers=N, capacity=CAPACITY, M1=M1, M2=M2,
        M_vehicle=M_VEHICLE,
        total_demand=float(D[1:N + 1].sum()),
        K_min=int(np.ceil(D[1:N + 1].sum() / CAPACITY)),
        K_range=[int(k) for k in sorted(K_sens.keys(), key=int)],
        best_K=int(best_K),
        best=dict(
            K=int(best_K),
            routes=best_routes_with_depot,
            total_travel_time=float(travel),
            total_tw_penalty=float(penalty),
            J_inner=float(J_inner),
            objective_M=float(obj_M),
            n_violators=int(n_violators_total),
        ),
        per_vehicle=schedule_per_vehicle,
        K_sensitivity={int(k): dict(travel=v["travel"], penalty=v["penalty"],
                                    J_inner=v["J_inner"], obj_M=v["obj_M"],
                                    n_routes=len(v["routes"]))
                       for k, v in K_sens.items()},
        comparison_with_reference=data["comparison_with_main_py"],
        note=("综合目标 J = M·K + travel + penalty （M=1000）。"
              "K* = 7 时综合目标 7149，与 main.py 已知参考解持平，"
              "并在 K=5/6/8/9 全部优于参考解，按铁律 §二.1 标'近似最优解'。"),
    )
    out_json = OUT_RESULT / "q4_pure_python.json"
    out_json.write_text(json.dumps(final_doc, ensure_ascii=False, indent=2, default=_to_jsonable),
                        encoding="utf-8")
    print(f"[写出] {out_json.relative_to(ROOT)}")

    # ---- K 灵敏度 JSON ----
    sens_doc = dict(
        K_range=[int(k) for k in sorted(K_sens.keys(), key=int)],
        results={int(k): dict(travel=v["travel"], penalty=v["penalty"],
                              J_inner=v["J_inner"], obj_M=v["obj_M"])
                 for k, v in K_sens.items()},
        best_K=int(best_K),
        comparison_with_reference=data["comparison_with_main_py"],
    )
    (OUT_SENS / "q4_K_sensitivity.json").write_text(
        json.dumps(sens_doc, ensure_ascii=False, indent=2, default=_to_jsonable),
        encoding="utf-8")
    print(f"[写出] {(OUT_SENS / 'q4_K_sensitivity.json').relative_to(ROOT)}")

    # ---- 表 1：路线表 ----
    rows = []
    for v in schedule_per_vehicle:
        for r in v["schedule"]:
            rows.append(dict(vehicle=v["vehicle"], **r))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE / "tab_04_q4_routes.csv", index=False, encoding="utf-8-sig")
    tex = ("\\begin{table}[htbp]\n\\centering\n"
           f"\\caption{{问题 4 最优 K^*={best_K} 多车辆调度（综合目标 J={obj_M:.0f}）}}\\label{{tab:q4_routes}}\n"
           "\\small\n\\begin{tabular}{cccccccc}\n\\toprule\n"
           "车辆 & 客户 & 到达 & 时间窗 & 早到 & 晚到 & 惩罚 & 离开 \\\\\n\\midrule\n")
    for v in schedule_per_vehicle:
        for r in v["schedule"]:
            tex += (f"{v['vehicle']} & {r['customer']} & {r['arrive']:.0f} & "
                    f"[{r['tw_a']:.0f},{r['tw_b']:.0f}] & "
                    f"{r['early']:.0f} & {r['late']:.0f} & "
                    f"{r['penalty']:.0f} & {r['depart']:.0f} \\\\\n")
        tex += (f"\\multicolumn{{2}}{{r}}{{车 {v['vehicle']} 小计}} & "
                f"travel={v['travel']:.0f} & demand={v['demand']:.0f}/{CAPACITY} & "
                f"& pen={v['penalty']:.0f} & \\\\\n")
        tex += "\\midrule\n"
    tex += (f"\\multicolumn{{6}}{{r}}{{总 travel}} & {travel:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{总 penalty}} & {penalty:.0f} & \\\\\n"
            f"\\multicolumn{{6}}{{r}}{{综合目标 1000K+travel+pen}} & {obj_M:.0f} & \\\\\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (OUT_TABLE / "tab_04_q4_routes.tex").write_text(tex, encoding="utf-8")
    print(f"[写出] tables/tab_04_q4_routes.csv + .tex")

    # ---- 表 2：K 灵敏度（含 vs 参考对比） ----
    REF_OBJ = data["comparison_with_main_py"]["REF_OBJ"]
    REF_TR = data["comparison_with_main_py"]["REF_TR"]
    REF_PE = data["comparison_with_main_py"]["REF_PE"]
    Ks_sorted = sorted([int(k) for k in K_sens.keys()])
    df_sens = pd.DataFrame([
        dict(K=k,
             mine_travel=K_sens[str(k)]["travel"],
             mine_penalty=K_sens[str(k)]["penalty"],
             mine_obj=K_sens[str(k)]["obj_M"],
             ref_travel=REF_TR[str(k)],
             ref_penalty=REF_PE[str(k)],
             ref_obj=REF_OBJ[str(k)],
             delta=K_sens[str(k)]["obj_M"] - REF_OBJ[str(k)])
        for k in Ks_sorted
    ])
    df_sens.to_csv(OUT_TABLE / "tab_04_q4_K_sensitivity.csv", index=False, encoding="utf-8-sig")
    tex2 = ("\\begin{table}[htbp]\n\\centering\n"
            "\\caption{问题 4 · 车辆数 K 灵敏度对比（本文 vs 参考）}\\label{tab:q4_K_sens}\n"
            "\\small\n\\begin{tabular}{ccccccccc}\n\\toprule\n"
            "K & 本文 travel & 本文 pen & \\textbf{本文 obj} & 参考 travel & 参考 pen & 参考 obj & $\\Delta$ obj & 状态 \\\\\n\\midrule\n")
    for _, row in df_sens.iterrows():
        d = row["delta"]
        status = "我赢" if d < 0 else ("打平" if d == 0 else "参考赢")
        tex2 += (f"{int(row['K'])} & {row['mine_travel']:.0f} & {row['mine_penalty']:.0f} & "
                 f"\\textbf{{{row['mine_obj']:.0f}}} & "
                 f"{row['ref_travel']} & {row['ref_penalty']} & {row['ref_obj']} & "
                 f"{d:+.0f} & {status} \\\\\n")
    tex2 += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (OUT_TABLE / "tab_04_q4_K_sensitivity.tex").write_text(tex2, encoding="utf-8")
    print(f"[写出] tables/tab_04_q4_K_sensitivity.csv + .tex")

    # ---- 图 1：多车路径图 ----
    fig, ax = plt.subplots(figsize=(11, 11))
    K_total = len(routes)
    cmap = plt.colormaps.get_cmap("tab10")
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pos = {0: (0.0, 0.0)}
    for i in range(1, N + 1):
        pos[i] = (np.cos(angles[i - 1]) * 1.3, np.sin(angles[i - 1]) * 1.3)
    ax.scatter(0, 0, s=400, c="#444", marker="s", edgecolor="black", zorder=5)
    ax.text(0, 0, "0", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    for k_idx, r in enumerate(routes):
        if not r: continue
        color = cmap(k_idx % 10)
        full = [0] + r + [0]
        for a, b in zip(full[:-1], full[1:]):
            xa, ya = pos[a]; xb, yb = pos[b]
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, alpha=0.85))
        first_c = r[0]
        x, y = pos[first_c]
        ax.text(x * 1.12, y * 1.12, f"V{k_idx+1}", color=color, fontsize=12, fontweight="bold")
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
                 f"travel={travel:.0f}, penalty={penalty:.0f}, "
                 f"综合目标 J={obj_M:.0f}, 违反 {n_violators_total}/{N}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_routes.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_routes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_routes.png + .pdf")

    # ---- 图 2：多车甘特 ----
    fig, ax = plt.subplots(figsize=(13, 0.55 * sum(len(v["schedule"]) for v in schedule_per_vehicle) + 2))
    y_offset = 0; yticks = []; ylabels = []
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
    ax.set_title(f"Q4 · K^*={best_K} 多车辆甘特（绿/彩=未违反 红=违反），综合目标 J={obj_M:.0f}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_gantt.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_gantt.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_gantt.png + .pdf")

    # ---- 图 3：违反分布 ----
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
                 f"违反客户 {n_violators_total}/{N}, 总惩罚 = {penalty:.0f}, 综合目标 J = {obj_M:.0f}")
    ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
    if pen_vals and max(pen_vals) > 0:
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
    print(f"[写出] figures/fig_04_q4_violation.png + .pdf")

    # ---- 图 4：K 灵敏度（重做，含 vs 参考对比）----
    Ks = sorted([int(k) for k in K_sens.keys()])
    mine_obj = [K_sens[str(k)]["obj_M"] for k in Ks]
    ref_obj_list = [REF_OBJ[str(k)] for k in Ks]
    mine_travel = [K_sens[str(k)]["travel"] for k in Ks]
    mine_pen = [K_sens[str(k)]["penalty"] for k in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # 左：综合目标对比
    ax = axes[0]
    ax.plot(Ks, mine_obj, "o-", color="#3a78c2", lw=2, ms=10, label="本文")
    ax.plot(Ks, ref_obj_list, "s--", color="#d9534f", lw=1.5, ms=8, label="已知参考")
    for x_, y_ in zip(Ks, mine_obj):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold", color="#3a78c2")
    for x_, y_ in zip(Ks, ref_obj_list):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, fontweight="bold", color="#d9534f")
    ax.axvline(best_K, color="#5cb85c", lw=1.2, ls=":", alpha=0.7)
    ax.text(best_K, max(ref_obj_list) * 0.98, f"K^*={best_K}", ha="center", fontsize=10, color="#5cb85c", fontweight="bold")
    ax.set_xlabel("可用车辆数 K"); ax.set_ylabel("综合目标 = 1000·K + travel + pen")
    ax.set_xticks(Ks)
    ax.set_title("(a) 综合目标 K 灵敏度（vs 参考）")
    ax.grid(alpha=0.3); ax.legend()

    # 右：travel/penalty 分解
    ax = axes[1]
    ax2 = ax.twinx()
    l1, = ax.plot(Ks, mine_travel, "s-", color="#5cb85c", lw=1.6, ms=8, label="travel")
    l2, = ax2.plot(Ks, mine_pen, "^-", color="#d9534f", lw=1.6, ms=8, label="penalty")
    ax.set_xlabel("可用车辆数 K")
    ax.set_ylabel("travel", color="#5cb85c")
    ax2.set_ylabel("penalty", color="#d9534f")
    ax2.set_yscale("symlog")
    ax.set_xticks(Ks)
    ax.set_title("(b) travel & penalty 分解：K↑→travel↑, penalty↓")
    ax.grid(alpha=0.3)
    ax.legend(handles=[l1, l2], loc="upper center")

    fig.suptitle(f"Q4 · K 灵敏度分析（题目第 (iv) 问要求图）—— K^* = {best_K}, 综合目标 = {obj_M:.0f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_04_q4_K_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG / "fig_04_q4_K_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] figures/fig_04_q4_K_sensitivity.png + .pdf")

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
