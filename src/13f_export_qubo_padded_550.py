"""
把 12 个 QUBO 矩阵 padding 到 550×550（CIM CPQC-550 平台上传硬性要求）。
顺序：先 8-bit 精度调整 → 再 padding 0 到 550×550 → 输出 CSV。

数学等价性：原始 n×n QUBO 与 padded 550×550 QUBO 的最优解前 n 位完全相同；
padded 变量是孤立变量（无能量耦合），值任意，不影响目标。

输出
  results/基础模型/qubo_matrices/csv_8bit_550/<name>_dense_550.csv  整数 550×550 矩阵
  results/基础模型/qubo_matrices/csv_8bit_550/all_qubos_padded_index.csv 索引 + 验证表
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
OUT = ROOT / "results/基础模型/qubo_matrices/csv_8bit_550"
OUT.mkdir(parents=True, exist_ok=True)

CIM_QUBIT = 550  # CPQC-550 固定尺寸

kw.license.init(user_id="147786085198512130", sdk_code="ff5sqQ83fV2TtaCZhCHNJQReHBgmd0")
metadata = json.loads((SRC / "metadata.json").read_text(encoding="utf-8"))
print(f"导出 {len(metadata)} 个 QUBO 矩阵 padding 到 {CIM_QUBIT}×{CIM_QUBIT} ...\n")

index_rows = []
for m in metadata:
    name = m["name"]
    Q_raw = np.load(SRC / f"{name}.npy")
    nvar = Q_raw.shape[0]
    assert nvar <= CIM_QUBIT, f"{name} nvar={nvar} > {CIM_QUBIT}（不该出现）"

    # 步骤 1：8-bit 精度调整（必须在 padding 之前做）
    Q_8bit = kw.qubo.adjust_qubo_matrix_precision(Q_raw, bit_width=8)

    # 步骤 2：padding 到 550×550，右下角全 0
    Q_padded = np.zeros((CIM_QUBIT, CIM_QUBIT), dtype=int)
    Q_padded[:nvar, :nvar] = Q_8bit.astype(int)

    # 数学等价性自检：前 nvar 行/列与 8-bit 矩阵完全一致；其余全 0
    assert np.array_equal(Q_padded[:nvar, :nvar], Q_8bit.astype(int)), "上块不一致"
    assert np.all(Q_padded[nvar:, :] == 0), "下块未清零"
    assert np.all(Q_padded[:, nvar:] == 0), "右块未清零"
    pad_count = CIM_QUBIT - nvar  # 孤立变量数

    # 落盘 CSV（纯整数，逗号分隔，无 header）
    dense_path = OUT / f"{name}_dense_550.csv"
    np.savetxt(dense_path, Q_padded, delimiter=",", fmt="%d")
    size_kb = dense_path.stat().st_size / 1024

    index_rows.append(dict(
        name=name, problem=m["problem"],
        original_n_qubo_vars=nvar,
        padded_to=CIM_QUBIT,
        n_padded_isolated_vars=pad_count,
        bit8_min=int(Q_8bit.min()), bit8_max=int(Q_8bit.max()),
        nnz_after_padding=int(np.count_nonzero(Q_padded)),
        file=f"{name}_dense_550.csv",
        size_KB=round(size_kb, 1),
        purpose=m["purpose"],
        verification="前 nvar×nvar 与 8-bit 一致 ✓; 其余全 0 ✓",
    ))
    print(f"  [{name}] {m['problem']}: nvar={nvar} → padded {CIM_QUBIT}×{CIM_QUBIT}, "
          f"孤立变量 {pad_count} 个, dense_550.csv = {size_kb:.0f} KB")

df = pd.DataFrame(index_rows)
df.to_csv(OUT / "all_qubos_padded_index.csv", index=False, encoding="utf-8-sig")

print(f"\n[写出]")
print(f"  目录：{OUT.relative_to(ROOT)}")
print(f"  - 12 个 *_dense_550.csv（每个 550×550 整数矩阵 ≈ 1.5-2 MB）")
print(f"  - all_qubos_padded_index.csv（索引 + 等价性自检）")
print(f"\n上传方式：登录 platform.qboson.com → 提交任务 → 选择 CPQC-550")
print(f"          → 上传 *_dense_550.csv → 等待求解 → 下载 spin 解 (长度 551)")
print(f"          → 解码：取 spin[:nvar]，丢弃 spin[nvar:550]（孤立变量）+ spin[550] (辅助自旋)")
print(f"\n注意：每次上传 1 个矩阵 = 消耗 1 次 550 比特配额。配额按 task 计费，与矩阵实际密度无关。")
