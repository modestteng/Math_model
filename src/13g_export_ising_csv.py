"""
导出 12 个 QUBO 对应的 Ising 矩阵 CSV（8-bit 整数版本）。
用途：CIM 真机平台手动上传时，若选 "Ising" 模式则用这套 CSV。

流程：原始 Q → 8-bit 精度调整 → kw.conversion.qubo_matrix_to_ising_matrix → CSV
注意：转换会引入 1 个辅助自旋，所以 ising 维度 = QUBO 维度 + 1。

输出
  results/基础模型/qubo_matrices/csv_ising_8bit/<name>_ising.csv  整数 ising 矩阵
  results/基础模型/qubo_matrices/csv_ising_8bit/all_ising_index.csv 索引
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
OUT = ROOT / "results/基础模型/qubo_matrices/csv_ising_8bit"
OUT.mkdir(parents=True, exist_ok=True)

kw.license.init(user_id="147786085198512130", sdk_code="ff5sqQ83fV2TtaCZhCHNJQReHBgmd0")
metadata = json.loads((SRC / "metadata.json").read_text(encoding="utf-8"))

print(f"导出 {len(metadata)} 个 QUBO 对应的 Ising CSV ...\n")
index_rows = []
for m in metadata:
    name = m["name"]
    Q_raw = np.load(SRC / f"{name}.npy")
    Q_8bit = kw.qubo.adjust_qubo_matrix_precision(Q_raw, bit_width=8)
    ising_mat, ising_bias = kw.conversion.qubo_matrix_to_ising_matrix(Q_8bit)
    n_ising = ising_mat.shape[0]
    assert n_ising <= 550, f"{name} ising {n_ising} > 550"

    # ising 矩阵转 int（应该已经是整数）
    ising_int = ising_mat.astype(int)

    # 落盘 CSV: 纯数值，逗号分隔，无 header
    dense_path = OUT / f"{name}_ising.csv"
    np.savetxt(dense_path, ising_int, delimiter=",", fmt="%d")

    nnz = int(np.count_nonzero(ising_int))
    val_min, val_max = int(ising_int.min()), int(ising_int.max())
    size_kb = dense_path.stat().st_size / 1024

    index_rows.append(dict(
        name=name, problem=m["problem"],
        n_qubo_vars=m["n_qubo_vars"],
        n_ising_spins=int(n_ising),
        ising_bias=float(ising_bias),
        within_cim_550=bool(n_ising <= 550),
        ising_min=val_min, ising_max=val_max,
        nnz=nnz,
        ising_csv=f"{name}_ising.csv",
        size_KB=round(size_kb, 1),
        purpose=m["purpose"],
    ))
    print(f"  [{name}] {m['problem']}: ising {n_ising}×{n_ising}, "
          f"val ∈ [{val_min}, {val_max}], nnz={nnz}, "
          f"bias={ising_bias:.1f}, size={size_kb:.1f} KB")

df = pd.DataFrame(index_rows)
df.to_csv(OUT / "all_ising_index.csv", index=False, encoding="utf-8-sig")

# 汇总文件大小
total_kb = sum(r["size_KB"] for r in index_rows)
print(f"\n[写出]")
print(f"  目录：{OUT.relative_to(ROOT)}")
print(f"  - 12 个 *_ising.csv（8-bit Ising 整数矩阵 → 平台 Ising 模式上传用）")
print(f"  - all_ising_index.csv（索引，含 ising_bias）")
print(f"  - 12 个文件总大小：{total_kb:.0f} KB ≈ {total_kb/1024:.1f} MB")
print(f"\n上传方式：")
print(f"  1. 登录 platform.qboson.com")
print(f"  2. 任务配置 → Quota 模式 + Ising 矩阵")
print(f"  3. 任务名：例如 all_4q_2026_04_26")
print(f"  4. 本地上传：选中全部 12 个 *_ising.csv（一次任务最多 20 个文件）")
print(f"  5. 提交（**消耗 1 次配额，并行求解全部 12 个矩阵**）")
print(f"  6. 等任务完成后下载 spin 解，每个矩阵的 spin[:nvar] 解码（最后 1 位辅助自旋丢弃）")
