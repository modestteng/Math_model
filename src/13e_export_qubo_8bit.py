"""
导出 8-bit 精度调整后的 QUBO 矩阵 CSV。
这是 CIM 真机实际处理的矩阵（kw.qubo.adjust_qubo_matrix_precision），
也是 platform.qboson.com 平台手动上传应使用的格式。

输出
  results/基础模型/qubo_matrices/csv_8bit/<name>_dense.csv  整数矩阵 (8-bit 缩放)
  results/基础模型/qubo_matrices/csv_8bit/<name>_coo.csv    稀疏 (i, j, value)
  results/基础模型/qubo_matrices/csv_8bit/all_qubos_8bit_index.csv 索引
"""
from __future__ import annotations
import os
for _k in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY",
           "all_proxy", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import kaiwu as kw

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/基础模型/qubo_matrices"
OUT = ROOT / "results/基础模型/qubo_matrices/csv_8bit"
OUT.mkdir(parents=True, exist_ok=True)

# Kaiwu 即使不调真机也需要 license（adjust_qubo_matrix_precision 是本地函数，但保险起见 init）
kw.license.init(user_id="147786085198512130", sdk_code="ff5sqQ83fV2TtaCZhCHNJQReHBgmd0")

metadata = json.loads((SRC / "metadata.json").read_text(encoding="utf-8"))
print(f"导出 {len(metadata)} 个 QUBO 矩阵的 8-bit 精度调整版本...\n")

index_rows = []
for m in metadata:
    name = m["name"]
    Q_raw = np.load(SRC / f"{name}.npy")
    Q_8bit = kw.qubo.adjust_qubo_matrix_precision(Q_raw, bit_width=8)
    nvar = Q_8bit.shape[0]

    # Dense CSV: 整数矩阵
    dense_path = OUT / f"{name}_dense.csv"
    np.savetxt(dense_path, Q_8bit, delimiter=",", fmt="%d")

    # COO Sparse CSV
    coo_rows = []
    for i in range(nvar):
        for j in range(i, nvar):
            v = Q_8bit[i, j]
            if v != 0:
                coo_rows.append((i, j, int(v)))
    coo_df = pd.DataFrame(coo_rows, columns=["row_i", "col_j", "value"])
    coo_path = OUT / f"{name}_coo.csv"
    coo_df.to_csv(coo_path, index=False, encoding="utf-8")

    raw_min, raw_max = float(Q_raw.min()), float(Q_raw.max())
    new_min, new_max = int(Q_8bit.min()), int(Q_8bit.max())
    nnz_8bit = int(np.count_nonzero(Q_8bit))
    dense_kb = dense_path.stat().st_size / 1024
    coo_kb = coo_path.stat().st_size / 1024

    index_rows.append(dict(
        name=name, problem=m["problem"],
        n_qubo_vars=nvar, n_ising_spins=m["n_ising_spins"],
        within_cim_550=m["within_cim_limit"],
        raw_min=raw_min, raw_max=raw_max,
        bit8_min=new_min, bit8_max=new_max,
        bit8_nnz=nnz_8bit,
        dense_csv=f"{name}_dense.csv",
        dense_size_KB=round(dense_kb, 1),
        coo_csv=f"{name}_coo.csv",
        coo_size_KB=round(coo_kb, 1),
        purpose=m["purpose"],
    ))
    print(f"  [{name}] {m['problem']}: raw [{raw_min:.0f},{raw_max:.0f}] → "
          f"8bit [{new_min},{new_max}], nnz={nnz_8bit}, "
          f"dense={dense_kb:.1f}KB")

df = pd.DataFrame(index_rows)
df.to_csv(OUT / "all_qubos_8bit_index.csv", index=False, encoding="utf-8-sig")

print(f"\n[写出]")
print(f"  目录：{OUT.relative_to(ROOT)}")
print(f"  - 12 个 *_dense.csv（8-bit 整数矩阵 → CIM 平台上传用）")
print(f"  - 12 个 *_coo.csv（8-bit 稀疏三元组）")
print(f"  - all_qubos_8bit_index.csv（索引 + 缩放对照）")
print(f"\n说明：CIM CPQC-550 真机接受 8-bit signed integer 矩阵。")
print(f"      原始 Q 矩阵 [-400, 400] 经 adjust_qubo_matrix_precision 缩放为整数小范围。")
