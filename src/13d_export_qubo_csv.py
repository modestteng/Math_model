"""
把 12 个 QUBO Q 矩阵从 .npy 转成 CSV 格式，方便：
  1) 手动上传到 Kaiwu 真机平台 platform.qboson.com
  2) 论文附录引用矩阵规模与样例数值

每个 Q 矩阵生成 2 种 CSV：
  ① <name>_dense.csv     ：n×n 完整矩阵（无标题行，纯数值，方便平台读取）
  ② <name>_coo.csv       ：稀疏 (i, j, value) 三元组（仅非零项，方便论文展示）
另外生成总览：
  all_qubos_index.csv     ：12 个矩阵的索引表
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/基础模型/qubo_matrices"
OUT = ROOT / "results/基础模型/qubo_matrices/csv"
OUT.mkdir(parents=True, exist_ok=True)

metadata = json.loads((SRC / "metadata.json").read_text(encoding="utf-8"))
print(f"导出 {len(metadata)} 个 QUBO 矩阵到 CSV...")

index_rows = []
for m in metadata:
    name = m["name"]
    Q = np.load(SRC / f"{name}.npy")
    nvar = Q.shape[0]

    # ① Dense CSV: n×n 完整矩阵，纯数值（无 header / index），平台直传
    dense_path = OUT / f"{name}_dense.csv"
    np.savetxt(dense_path, Q, delimiter=",", fmt="%g")

    # ② Sparse COO CSV: 仅非零元素（含对角）
    coo_rows = []
    for i in range(nvar):
        for j in range(i, nvar):
            v = Q[i, j]
            if v != 0:
                coo_rows.append((i, j, v))
    coo_df = pd.DataFrame(coo_rows, columns=["row_i", "col_j", "value"])
    coo_path = OUT / f"{name}_coo.csv"
    coo_df.to_csv(coo_path, index=False, encoding="utf-8")

    # 文件大小
    dense_kb = dense_path.stat().st_size / 1024
    coo_kb = coo_path.stat().st_size / 1024

    index_rows.append(dict(
        name=name, problem=m["problem"],
        n_qubo_vars=nvar, n_ising_spins=m["n_ising_spins"],
        within_cim_550=m["within_cim_limit"],
        nnz=m["nnz"],
        dense_csv=f"{name}_dense.csv",
        dense_size_KB=round(dense_kb, 1),
        coo_csv=f"{name}_coo.csv",
        coo_size_KB=round(coo_kb, 1),
        purpose=m["purpose"],
    ))
    print(f"  [{name}] {m['problem']}, nvar={nvar}, "
          f"dense={dense_kb:.1f}KB, coo={coo_kb:.1f}KB ({m['nnz']} 非零)")

# 索引表
df_idx = pd.DataFrame(index_rows)
df_idx.to_csv(OUT / "all_qubos_index.csv", index=False, encoding="utf-8-sig")

print(f"\n[写出]")
print(f"  目录：{OUT.relative_to(ROOT)}")
print(f"  - 12 个 *_dense.csv（完整矩阵，平台上传用）")
print(f"  - 12 个 *_coo.csv（稀疏三元组，论文附录用）")
print(f"  - all_qubos_index.csv（索引表）")
print(f"\n建议：上传 Kaiwu 平台时用 *_dense.csv（确认平台具体格式后可二次导出）")
