"""
从中断的 CIM 真机 stdout 抢救 Q1/Q2 已完成的结果，落盘到 JSON。
脚本运行于 2026-04-26 19:51:03，Q1/Q2 已完成、Q3-seg1 已提交但结果未取回（脚本 kill）。

来源：stdout 文件（包含 task_name + 完成时间 + 可行解数 + 最低 H + 最优 travel/J）
注：Q1/Q2 的 10 个样本 spin 矩阵已丢失，仅保留汇总数值。
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/基础模型"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 从 stdout 抢救的数值 ----
TIMESTAMP = "20260426_195103"

q1_result = dict(
    problem="Q1 · n=15 单车 TSP",
    cim_task_name=f"q1_n15_{TIMESTAMP}",
    cim_machine="CPQC-550",
    cim_submit_time="2026-04-26 19:51:07",
    cim_complete_time="2026-04-26 19:52:07",
    cim_solve_time_sec=63.4,
    cim_sample_number=10,
    n_qubo_vars=225,
    n_ising_spins=226,
    bit_width_8_applied=True,
    encoding="one-hot 位置编码 + distance（同方案 C，无时间窗）",
    n_total_samples=10,
    n_feasible=1,
    feasibility_rate=0.10,
    min_hamiltonian=-800.0,
    best_travel=31.0,
    py_baseline_travel=29.0,  # Held-Karp 精确最优
    delta_vs_py=2.0,
    delta_pct="+6.9%",
    note=(
        "CIM 真机 1/10 可行（10%），polish 后 travel=31（差 Held-Karp 精确解 +2，6.9%）。"
        "本次为 R-Q1-CIM-002 重复验证，与 R-Q1-004 (2026-04-26 16:59) 的 cost=31 一致。"
        "10 个样本的 spin 二进制矩阵因脚本中断未落盘，仅保留汇总数值。"
    ),
    status="completed_and_recovered",
)
(OUT / "q1_cim_real_v2.json").write_text(
    json.dumps(q1_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {(OUT / 'q1_cim_real_v2.json').relative_to(ROOT)}")

q2_result = dict(
    problem="Q2 · n=15 单车 + 时间窗（方案 C：QUBO 同 Q1，时间窗在解码后评估）",
    cim_task_name=f"q2_n15_{TIMESTAMP}",
    cim_machine="CPQC-550",
    cim_submit_time="2026-04-26 19:52:09",
    cim_complete_time="2026-04-26 19:54:10",
    cim_solve_time_sec=122.3,
    cim_sample_number=10,
    n_qubo_vars=225,
    n_ising_spins=226,
    bit_width_8_applied=True,
    encoding="one-hot 位置编码 + distance（方案 C），时间窗在解码后用 polish 优化",
    n_total_samples=10,
    n_feasible=1,
    feasibility_rate=0.10,
    min_hamiltonian=-800.0,
    best_travel=31.0,
    best_penalty=84090.0,
    best_J=84121.0,
    py_baseline_J=84121.0,
    delta_vs_py=0.0,
    delta_pct="+0.00%",
    matches_pure_python="True",
    note=(
        "CIM 真机 1/10 可行（10%），polish 后 J = 84121 与纯 Python v2 完全一致。"
        "构成跨求解器（CIM 真机 vs 纯 Python LNS）独立收敛同一解的稳健性证据。"
        "10 个样本的 spin 矩阵因脚本中断未落盘，仅保留汇总数值。"
    ),
    status="completed_and_recovered",
)
(OUT / "q2_cim_real.json").write_text(
    json.dumps(q2_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {(OUT / 'q2_cim_real.json').relative_to(ROOT)}")

q3_seg1_result = dict(
    problem="Q3 · n=50 第 1 段（17 客户 子 QUBO 289 比特）",
    cim_task_name=f"q3_seg1_{TIMESTAMP}",
    cim_machine="CPQC-550",
    cim_submit_time="2026-04-26 19:54:12",
    cim_complete_time="未知（脚本 kill 时还在 'still processing'）",
    cim_sample_number=10,
    n_qubo_vars=289,
    n_ising_spins=290,
    bit_width_8_applied=True,
    n_feasible="未知（脚本 kill 时未取回）",
    status="quota_consumed_but_result_lost",
    note=(
        "task 已提交 CIM 服务端（配额已扣 1 次），但 Python 脚本因配额提示被 kill，"
        "wait=True 阻塞过程中断；CIM 任务结果在服务器队列中可能仍在或已完成，"
        "可通过 task_name='q3_seg1_20260426_195103' 在 Kaiwu 平台 https://platform.qboson.com 查询历史任务。"
        "本次实验视为该子 QUBO 配额消耗但结果丢失。"
    ),
    recovery_action_required=(
        "登录 platform.qboson.com → 任务列表 → 找 q3_seg1_20260426_195103 → "
        "若状态=completed 可下载 spin 文件手动解码；否则视作浪费配额。"
    ),
)
(OUT / "q3_seg1_cim_real_lost.json").write_text(
    json.dumps(q3_seg1_result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {(OUT / 'q3_seg1_cim_real_lost.json').relative_to(ROOT)}")

# 汇总
summary = dict(
    timestamp=TIMESTAMP,
    cim_attempts=[
        dict(name="q1_n15", status="✓ 完成并已抢救", best="travel=31"),
        dict(name="q2_n15", status="✓ 完成并已抢救", best="J=84121 (与 Python 一致)"),
        dict(name="q3_seg1", status="✗ 配额已扣但结果丢失", action="可去 platform.qboson.com 查"),
    ],
    quota_consumed=3,
    quota_remaining_estimate="6 - 3 = 3",
    next_steps=(
        "用剩余 3 次配额跑：① Q3-seg2 ② Q3-seg3 ③ Q4 任选 1 车（推荐 q4_v3 = 64 比特最大）"
    ),
)
(OUT / "cim_recovery_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[写出] {(OUT / 'cim_recovery_summary.json').relative_to(ROOT)}")
print("\n=== 抢救完成 ===")
print(f"  Q1: travel = 31  (差 Held-Karp 全局最优 +2)")
print(f"  Q2: J = 84121     (与纯 Python 完全一致 ✓)")
print(f"  Q3-seg1: 配额扣 1 次但结果丢失（可去 qboson 平台查 task='q3_seg1_{TIMESTAMP}'）")
print(f"  剩余配额估计：3 次")
