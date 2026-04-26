"""
导出全 4 问的 12 个 QUBO Q 矩阵到磁盘（不调用 CIM，零配额消耗）。

用途：
  1) 让用户查看每个 Q 矩阵的尺寸与数值
  2) 后续单独提交某个矩阵到 CIM（节省配额）
  3) 论文附录引用矩阵规模与构造细节

落盘
  results/基础模型/qubo_matrices/q1.npy, q2.npy
  results/基础模型/qubo_matrices/q3_seg1.npy, q3_seg2.npy, q3_seg3.npy
  results/基础模型/qubo_matrices/q4_v1..v7.npy
  results/基础模型/qubo_matrices/metadata.json   # 全部 12 个矩阵的元信息
  results/基础模型/qubo_matrices/matrix_summary.csv  # 比特数 / 客户 / 用途等
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT = ROOT / "results/基础模型/qubo_matrices"
OUT.mkdir(parents=True, exist_ok=True)

A_PEN = 200.0

# ---- 数据 ----
nodes_raw = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
nodes_raw.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
T_FULL = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int).astype(float)


def build_qubo_tsp(T_sub: np.ndarray, n_sub: int, A_pen: float = A_PEN) -> np.ndarray:
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


def make_sub_T(a: int, customers: list[int], b: int) -> np.ndarray:
    k = len(customers)
    T = np.zeros((k + 1, k + 1))
    for i in range(1, k + 1):
        ci = customers[i - 1]
        T[0, i] = T_FULL[a, ci]
        T[i, 0] = T_FULL[ci, b]
        for j in range(1, k + 1):
            if i != j: T[i, j] = T_FULL[ci, customers[j - 1]]
    return T


# ---- 12 个 QUBO 配置 ----
PY_Q3_PERM = [40, 2, 21, 26, 12, 28, 27, 1, 31, 7, 19, 48, 8, 18, 5, 6, 37, 42,
              15, 43, 14, 38, 44, 16, 17, 45, 46, 47, 36, 49, 11, 10, 30, 32,
              20, 9, 34, 35, 33, 50, 3, 29, 24, 25, 4, 39, 23, 22, 41, 13]
PY_Q4_ROUTES = [
    [15, 2, 21, 40, 6, 37, 17],
    [31, 30, 9, 34, 35, 20, 32],
    [27, 16, 44, 38, 14, 43, 42, 13],
    [47, 19, 36, 49, 11, 10, 1],
    [33, 25, 39, 23, 22, 41, 4],
    [28, 29, 12, 3, 50, 26, 24],
    [5, 45, 7, 18, 8, 46, 48],
]

specs = []

# Q1 / Q2 共用同一矩阵（n=15, 0..0），但分开落盘以方便区分用途
specs.append(dict(
    name="q1", problem="Q1", n_sub=15, a_node=0, b_node=0,
    customers=list(range(1, 16)),
    purpose="Q1: n=15 单车 TSP，QUBO 仅含距离 + one-hot 罚",
    py_baseline="travel = 29 (Held-Karp 全局最优)",
))
specs.append(dict(
    name="q2", problem="Q2", n_sub=15, a_node=0, b_node=0,
    customers=list(range(1, 16)),
    purpose="Q2: n=15 单车 + 时间窗。方案 C：QUBO 同 Q1 结构（仅 one-hot+距离），时间窗在解码后评估",
    py_baseline="J = travel + tw_pen = 84121",
))
# Q3 三段
seg_bounds = [(0, 17), (17, 34), (34, 50)]
for i, (lo, hi) in enumerate(seg_bounds):
    a = 0 if lo == 0 else PY_Q3_PERM[lo - 1]
    b = 0 if hi == 50 else PY_Q3_PERM[hi]
    specs.append(dict(
        name=f"q3_seg{i+1}", problem="Q3", n_sub=hi-lo, a_node=a, b_node=b,
        customers=PY_Q3_PERM[lo:hi],
        purpose=f"Q3: n=50 滚动窗口分解第 {i+1}/3 段（段前节点 {a} → 段后节点 {b}）",
        py_baseline="（段内）；Q3 全局 J=4941906",
    ))
# Q4 七车
for i, customers in enumerate(PY_Q4_ROUTES):
    specs.append(dict(
        name=f"q4_v{i+1}", problem="Q4", n_sub=len(customers), a_node=0, b_node=0,
        customers=customers,
        purpose=f"Q4: 多车 K=7 第 {i+1}/7 辆车（depot → 客户 → depot）",
        py_baseline="（车内）；Q4 全局 obj_M=7149",
    ))

# ---- 构造 + 落盘 ----
metadata = []
print(f"构造并导出 {len(specs)} 个 QUBO Q 矩阵...")
for s in specs:
    T_sub = make_sub_T(s["a_node"], s["customers"], s["b_node"])
    Q = build_qubo_tsp(T_sub, s["n_sub"], A_PEN)
    nvar = Q.shape[0]
    n_ising = nvar + 1  # 含辅助自旋

    # 落盘：原始 Q 矩阵（双精度浮点）
    np.save(OUT / f"{s['name']}.npy", Q)
    # 也落盘子 T（方便复算）
    np.save(OUT / f"{s['name']}_T_sub.npy", T_sub)

    nnz = int(np.count_nonzero(Q))
    diag_nnz = int(np.count_nonzero(np.diag(Q)))
    upper_nnz = int(np.count_nonzero(np.triu(Q, k=1)))
    val_min = float(Q.min()); val_max = float(Q.max())
    val_min_offdiag = float(Q[~np.eye(nvar, dtype=bool)].min())
    val_max_offdiag = float(Q[~np.eye(nvar, dtype=bool)].max())

    item = dict(
        name=s["name"], problem=s["problem"],
        n_sub=s["n_sub"],
        n_qubo_vars=int(nvar),
        n_ising_spins=int(n_ising),
        cim_qubit_limit=550,
        within_cim_limit=bool(n_ising <= 550),
        a_node=s["a_node"], b_node=s["b_node"],
        customers=[int(c) for c in s["customers"]],
        purpose=s["purpose"],
        py_baseline=s["py_baseline"],
        A_pen=A_PEN,
        encoding="one-hot 位置编码：x[i,p]，i=客户索引，p=访问位置；nvar = n_sub²",
        nnz=nnz,
        diag_nnz=diag_nnz,
        upper_nnz=upper_nnz,
        value_min=val_min, value_max=val_max,
        value_min_offdiag=val_min_offdiag, value_max_offdiag=val_max_offdiag,
        files=dict(
            qubo_matrix_npy=f"{s['name']}.npy",
            sub_T_matrix_npy=f"{s['name']}_T_sub.npy",
        ),
    )
    metadata.append(item)
    print(f"  [{s['name']}] {s['problem']}, n_sub={s['n_sub']}, nvar={nvar}, "
          f"ising={n_ising} (≤550 ✓), nnz={nnz}, val ∈ [{val_min:.0f},{val_max:.0f}]")

(OUT / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# CSV 摘要表
df = pd.DataFrame([
    dict(name=m["name"], problem=m["problem"],
         n_sub=m["n_sub"], n_qubo_vars=m["n_qubo_vars"],
         n_ising=m["n_ising_spins"], within_cim=m["within_cim_limit"],
         nnz=m["nnz"], val_min=m["value_min"], val_max=m["value_max"],
         purpose=m["purpose"])
    for m in metadata
])
df.to_csv(OUT / "matrix_summary.csv", index=False, encoding="utf-8-sig")

print(f"\n[写出]")
print(f"  目录：{OUT.relative_to(ROOT)}")
print(f"  - {len(metadata)} 个 .npy 矩阵 + 对应的 sub_T .npy")
print(f"  - metadata.json (全部元信息)")
print(f"  - matrix_summary.csv (摘要表)")
print(f"\n比特预算汇总：")
print(df[["name", "problem", "n_qubo_vars", "n_ising", "within_cim"]].to_string(index=False))
