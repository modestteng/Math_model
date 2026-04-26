"""
问题 1 · Kaiwu CIM 真机验证（玻色相干光量子计算机 · v2 改进模型）

[WARN] 本脚本会通过云端任务队列把 Ising 矩阵提交到真实的相干光 CIM 求解器，
   消耗 SDK 帐号下的真实 quota / sample 配额。请在确认 KAIWU_USER_ID
   与 KAIWU_SDK_CODE 正确、且配额充足后再运行。

环境：D:/Anaconda/envs/kaiwu-py310/python.exe   (Python 3.10)
运行：D:/Anaconda/envs/kaiwu-py310/python.exe src/04_q1_kaiwu_cim_real.py

流程：
  1. license 初始化
  2. _q_lib.build_qubo_q1()       → (225, 225) 上三角 QUBO  (A=200)
  3. adjust_qubo_matrix_precision → 8-bit 量化 QUBO (CIM 硬件位宽要求)
  4. qubo_matrix_to_ising_matrix  → (226, 226) Ising + bias  （含辅助 spin）
  5. CIMOptimizer(sample_number=10, wait=True) 提交云端任务，阻塞等待结果
  6. spin → binary，复算 QUBO 值，hybrid_polish 后处理
  7. kw.common.hamiltonian 复算每个样本的哈密顿量 → 落盘曲线图
  8. 与 SDK SA 基线 / 加强版结果对比，写入 results/改进模型/qubo_v2_q1_cim_real.json
"""
from __future__ import annotations
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 仅在本进程内临时清除代理变量，不修改用户系统代理
for _v in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)

import kaiwu as kw  # type: ignore

from _q_lib import (
    load_data,
    build_qubo_q1,
    to_symmetric,
    decode,
    route_cost,
    hybrid_polish,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_RESULT = ROOT / "results/改进模型"
OUT_FIG = ROOT / "figures"
for p in (OUT_RESULT, OUT_FIG):
    p.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 配置
# ------------------------------------------------------------
N = 15
A_PEN = 200.0
SEED = 20260426

KAIWU_USER_ID = "147786085198512130"
KAIWU_SDK_CODE = "ff5sqQ83fV2TtaCZhCHNJQReHBgmd0"

# CIM 真机参数
CIM_TASK_MODE = "quota"      # 'quota' or 'sample' — 配额池不同，先试 quota
CIM_SAMPLE_NUMBER = 10       # 仅 sample 模式生效；最小 10
CIM_WAIT = True              # 同步阻塞等待
CIM_INTERVAL = 1             # 异步轮询时使用，分钟


def spin_to_binary(spin_solutions: np.ndarray, nvar: int) -> np.ndarray:
    s = np.asarray(spin_solutions, dtype=np.int8)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if s.shape[1] == nvar + 1:
        s_aux = s[:, -1:]
        s_fixed = s * s_aux
        s_main = s_fixed[:, :-1]
    elif s.shape[1] == nvar:
        s_main = s
    else:
        raise ValueError(f"unexpected spin shape {s.shape}")
    return ((s_main + 1) // 2).astype(np.int8)


def main():
    print("=" * 64)
    print("  问题 1 · Kaiwu CIM 真机验证（v2 改进模型）")
    print("=" * 64)
    print(f"  n = {N}   A = {A_PEN}   sample_number = {CIM_SAMPLE_NUMBER}")
    print(f"  task_mode = {CIM_TASK_MODE}   wait = {CIM_WAIT}")
    print()

    # ---- 1) 鉴权 ----
    print("[1/8] license 初始化 ...")
    kw.license.init(user_id=KAIWU_USER_ID, sdk_code=KAIWU_SDK_CODE)
    print(f"      kaiwu = {kw.__version__}   user_id = {KAIWU_USER_ID}")

    # ---- 2) 构造 QUBO ----
    print("\n[2/8] 构造 QUBO ...")
    T_mat, _ = load_data(N)
    Q_upper = build_qubo_q1(T_mat, N, A=A_PEN)
    Q_full_orig = to_symmetric(Q_upper)
    nvar = Q_upper.shape[0]
    const_term = 2 * N * A_PEN
    print(f"      变量数 nvar = {nvar}   非零项 = {int(np.count_nonzero(Q_upper))}")
    print(f"      原始 QUBO 元素范围: [{Q_upper.min():.0f}, {Q_upper.max():.0f}]")

    # ---- 3) 8-bit 精度调整（CIM 硬件位宽限制）----
    print("\n[3/8] 8-bit 精度调整 (kw.qubo.adjust_qubo_matrix_precision) ...")
    Q_adjusted = kw.qubo.adjust_qubo_matrix_precision(Q_upper, bit_width=8)
    Q_adjusted_full = to_symmetric(Q_adjusted)
    print(f"      量化后 QUBO 元素范围: [{Q_adjusted.min():.0f}, {Q_adjusted.max():.0f}]")
    try:
        kw.qubo.check_qubo_matrix_bit_width(Q_adjusted, bit_width=8)
        print("      [OK] 8-bit 校验通过")
    except ValueError as e:
        print(f"      [FAIL] 8-bit 校验失败: {e}")
        raise

    # ---- 4) QUBO → Ising ----
    print("\n[4/8] QUBO → Ising 转换 ...")
    ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_adjusted)
    print(f"      ising_mat.shape = {ising_mat.shape}   ising_bias = {ising_bias:.4f}")

    # ---- 5) CheckpointManager 配置 ----
    ckpt_dir = Path(tempfile.gettempdir()) / "kaiwu_cim_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    kw.common.CheckpointManager.save_dir = str(ckpt_dir)
    print(f"\n[5/8] CheckpointManager.save_dir = {ckpt_dir}")

    # ---- 6) 提交 CIM 真机任务 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"q1_tsp_n{N}_A{int(A_PEN)}_{timestamp}"
    print(f"\n[6/8] 提交 CIM 真机任务 ... task_name = {task_name}")
    print("      [WARN]  正在向云端 CIM 求解器提交任务，可能需要数分钟 ...")
    optimizer = kw.cim.CIMOptimizer(
        task_name=task_name,
        wait=CIM_WAIT,
        interval=CIM_INTERVAL,
        task_mode=CIM_TASK_MODE,
        sample_number=CIM_SAMPLE_NUMBER,
    )
    t0 = time.time()
    spins = optimizer.solve(ising_mat)
    t_cim = time.time() - t0
    if spins is None:
        print(f"      [FAIL] 任务尚未完成（wait={CIM_WAIT}）。")
        return
    if spins.ndim == 1:
        spins = spins.reshape(1, -1)
    print(f"      [OK] 任务完成   耗时 = {t_cim:.2f}s   返回样本数 = {spins.shape[0]}")
    print(f"      spin 解形状 = {spins.shape}   (含辅助 spin → 末列)")

    # ---- 7) 解码与评估 ----
    print("\n[7/8] 解码 / 复算 / 后处理 ...")
    x_all = spin_to_binary(spins, nvar)
    ham_curve = kw.common.hamiltonian(ising_mat, spins)   # 每个样本的 Ising 能量
    print(f"      Ising 哈密顿量范围: [{ham_curve.min():.2f}, {ham_curve.max():.2f}]")

    # 注意：用 *原始未量化* QUBO 评估真实路径代价（而非量化后的 Q_adjusted）
    records = []
    for k in range(x_all.shape[0]):
        x = x_all[k]
        q_val_orig = float(kw.qubo.calculate_qubo_value(Q_full_orig, 0.0, x))
        q_val_adj = float(kw.qubo.calculate_qubo_value(Q_adjusted_full,
                                                        ising_bias, x))
        perm, feasible = decode(x, N)
        cost = route_cost(perm, T_mat) if feasible else float("inf")
        records.append(dict(
            x=x, perm=perm, feasible=feasible, cost=cost,
            qubo_orig=q_val_orig, qubo_adjusted=q_val_adj,
            ising_h=float(ham_curve[k]),
        ))
    records.sort(key=lambda r: (not r["feasible"], r["qubo_adjusted"]))
    feas = [r for r in records if r["feasible"]]
    print(f"      可行样本数 = {len(feas)} / {len(records)}")

    if feas:
        best = feas[0]
        polished, n_iters = hybrid_polish(best["perm"], T_mat)
        polished_cost = route_cost(polished, T_mat)
        polished_route = [0] + polished + [0]
        print(f"      最佳可行样本 cost = {best['cost']:.0f}   "
              f"原始 QUBO = {best['qubo_orig']:.1f}")
        print(f"      + 2-opt/Or-opt 打磨: iters = {n_iters}  cost = {polished_cost:.0f}")
        print(f"      路径: {' -> '.join(map(str, polished_route))}")
    else:
        best = records[0]
        polished, n_iters = best["perm"], 0
        polished_cost = best["cost"] if best["feasible"] else float("inf")
        polished_route = [0] + polished + [0]
        print("      [警告] CIM 真机未返回可行样本（可对比量化精度损失影响）")

    # ---- 8) 哈密顿量分布图 ----
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    # 左：每个样本的能量条形图
    ax = axs[0]
    sample_idx = np.arange(len(records))
    h_arr = np.array([r["ising_h"] for r in records])
    feas_mask = np.array([r["feasible"] for r in records])
    bars = ax.bar(sample_idx, h_arr, color=["#2ca02c" if f else "#d62728" for f in feas_mask])
    ax.set_xlabel("样本编号")
    ax.set_ylabel("Ising 哈密顿量 H")
    ax.set_title(f"CIM 真机 {len(records)} 个样本能量分布")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", label="可行"),
        plt.Rectangle((0, 0), 1, 1, fc="#d62728", label="违约"),
    ])
    ax.grid(alpha=0.3)
    # 右：QUBO 原始值 vs 路径代价
    ax = axs[1]
    feas_recs = [r for r in records if r["feasible"]]
    if feas_recs:
        costs = [r["cost"] for r in feas_recs]
        qubos = [r["qubo_orig"] for r in feas_recs]
        ax.scatter(qubos, costs, s=80, c="#1f77b4", edgecolor="black", zorder=3)
        ax.set_xlabel("原始 QUBO 值（A=200，未量化）")
        ax.set_ylabel("路径代价")
        ax.set_title("可行样本：QUBO 值 vs 路径代价")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "无可行样本", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    fig.suptitle("玻色 CIM 相干光量子计算机 · 问题 1 真机验证")
    fig.tight_layout()
    fig_png = OUT_FIG / "fig_q1_cim_hamiltonian.png"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n[图]  {fig_png.relative_to(ROOT)}  (+ .pdf)")

    # ---- 9) 落盘 ----
    out = dict(
        problem="Q1 · Kaiwu CIM 真机验证（v2 改进模型）",
        kaiwu_version=kw.__version__,
        timestamp=timestamp,
        task_name=task_name,
        n_customers=N,
        n_qubo_vars=nvar,
        penalty_A=A_PEN,
        const_term=const_term,
        precision_adjustment=dict(
            bit_width=8,
            orig_range=[float(Q_upper.min()), float(Q_upper.max())],
            adjusted_range=[float(Q_adjusted.min()), float(Q_adjusted.max())],
        ),
        ising_bias=float(ising_bias),
        cim_params=dict(
            task_mode=CIM_TASK_MODE,
            sample_number=CIM_SAMPLE_NUMBER,
            wait=CIM_WAIT,
            interval=CIM_INTERVAL,
            time_sec=round(t_cim, 3),
        ),
        cim_result=dict(
            n_samples=int(spins.shape[0]),
            n_feasible=len(feas),
            feasibility_rate=round(len(feas) / max(len(records), 1), 4),
            ising_h_min=float(h_arr.min()),
            ising_h_max=float(h_arr.max()),
            ising_h_mean=float(h_arr.mean()),
            best_route_cost=best["cost"] if best["feasible"] else None,
            best_qubo_orig=best["qubo_orig"] if best["feasible"] else None,
            best_perm=best["perm"] if best["feasible"] else None,
            samples=[
                dict(idx=i, feasible=r["feasible"],
                     cost=r["cost"] if r["feasible"] else None,
                     ising_h=r["ising_h"],
                     qubo_orig=r["qubo_orig"],
                     qubo_adjusted=r["qubo_adjusted"])
                for i, r in enumerate(records)
            ],
        ),
        cim_plus_2opt=dict(
            two_opt_iters=int(n_iters),
            cost=polished_cost if polished_cost != float("inf") else None,
            route=polished_route,
        ),
        figure=str(fig_png.relative_to(ROOT)),
    )
    out_json = OUT_RESULT / "qubo_v2_q1_cim_real.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                    default=lambda o: o.tolist() if hasattr(o, "tolist") else o),
                        encoding="utf-8")
    print(f"[写出] {out_json.relative_to(ROOT)}")

    # ---- 10) 与 SDK SA 结果对比 ----
    print("\n" + "=" * 64)
    print("  对比汇总（Q1 · n=15 · A=200）")
    print("=" * 64)
    sdk_json = ROOT / "results/基础模型/qubo_v1_q1_kaiwu_sdk.json"
    if sdk_json.exists():
        sdk = json.loads(sdk_json.read_text(encoding="utf-8"))
        print(f"  [SDK SA 基线]    cost = {sdk['sdk_sa_baseline']['best_route_cost']:.0f}   "
              f"+2-opt = {sdk['sdk_sa_baseline_plus_2opt']['cost']:.0f}   "
              f"feas = {sdk['sdk_sa_baseline']['n_feasible']}/"
              f"{sdk['sdk_sa_baseline']['n_unique_solutions']}")
        if sdk.get("sdk_sa_tuned"):
            t = sdk["sdk_sa_tuned"]
            print(f"  [SDK SA 加强版]  cost = "
                  f"{t['best_route_cost']:.0f}   "
                  f"+2-opt = {sdk['sdk_sa_tuned_plus_2opt']['cost']:.0f}   "
                  f"feas = {t['n_feasible']}/{t['n_total_candidates']}")
    if best["feasible"]:
        print(f"  [CIM 真机]       cost = {best['cost']:.0f}   "
              f"+2-opt = {polished_cost:.0f}   "
              f"feas = {len(feas)}/{len(records)}")
    else:
        print(f"  [CIM 真机]       未返回可行样本   feas = 0/{len(records)}")
    print()
    print("== 完成 · CIM 真机验证 ==")


if __name__ == "__main__":
    main()
