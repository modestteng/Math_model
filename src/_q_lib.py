"""
共享建模库：QUBO 构造、解、解码、TSP 局部搜索
被 01_q1_qubo_tsp.py（自实现 SA）与 02_q1_kaiwu_solve.py（Kaiwu SDK）共用
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"


# ------------------------------------------------------------
# 数据加载
# ------------------------------------------------------------
def load_data(n_customers: int):
    """返回 (T, nodes_df)；T 为 (n+1)x(n+1) depot+客户 1..n 的旅行时间矩阵"""
    nodes = pd.read_excel(DATA_XLSX, sheet_name=0, header=0)
    nodes.columns = ["ID", "tw_a", "tw_b", "service", "demand", "_blank", "capacity"]
    T_full = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
    assert T_full.shape == (51, 51)
    T = T_full[: n_customers + 1, : n_customers + 1].astype(float)
    return T, nodes


# ------------------------------------------------------------
# QUBO 构造（问题 1：单车辆 TSP，无时间窗、无容量）
#   变量 x[i,p] (i=1..n 客户; p=1..n 位置), 共 n^2
#   能量 H = A * H_const + H_dist
# ------------------------------------------------------------
def idx(i: int, p: int, n: int) -> int:
    return (i - 1) * n + (p - 1)


def build_qubo_q1(T: np.ndarray, n: int, A: float = 200.0) -> np.ndarray:
    """返回 (n^2, n^2) 的严格上三角 QUBO 矩阵 Q
    E = sum_k Q[k,k] x_k + sum_{k<l} Q[k,l] x_k x_l
    """
    nvar = n * n
    Q = np.zeros((nvar, nvar))

    # 列约束：每个位置 p 恰好一个客户
    for p in range(1, n + 1):
        vs = [idx(i, p, n) for i in range(1, n + 1)]
        for k in vs:
            Q[k, k] += -A
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                Q[vs[a], vs[b]] += 2 * A

    # 行约束：每个客户 i 恰好出现在一个位置
    for i in range(1, n + 1):
        vs = [idx(i, p, n) for p in range(1, n + 1)]
        for k in vs:
            Q[k, k] += -A
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                Q[vs[a], vs[b]] += 2 * A

    # 距离项
    for i in range(1, n + 1):
        Q[idx(i, 1, n), idx(i, 1, n)] += T[0, i]
        Q[idx(i, n, n), idx(i, n, n)] += T[i, 0]
    for p in range(1, n):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                k1, k2 = idx(i, p, n), idx(j, p + 1, n)
                ku, kv = (k1, k2) if k1 < k2 else (k2, k1)
                Q[ku, kv] += T[i, j]
    return Q


def to_symmetric(Q: np.ndarray) -> np.ndarray:
    """上三角 Q → 对称 S，使 x^T S x 等于 Q 的能量定义"""
    diag = np.diag(Q).copy()
    upper = np.triu(Q, k=1)
    S = upper + upper.T
    np.fill_diagonal(S, diag)
    return S


# ------------------------------------------------------------
# 解码与路径代价
# ------------------------------------------------------------
def decode(x: np.ndarray, n: int) -> tuple[list[int], bool]:
    M = np.asarray(x).reshape(n, n)
    feasible = bool(np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1))
    perm = []
    for p in range(n):
        col = M[:, p]
        if col.sum() == 1:
            perm.append(int(np.argmax(col)) + 1)
        else:
            perm.append(int(np.argmax(col)) + 1)
    return perm, feasible


def route_cost(route_or_perm: list[int], T: np.ndarray) -> float:
    if route_or_perm and route_or_perm[0] == 0:
        full = list(route_or_perm)
    else:
        full = [0] + list(route_or_perm) + [0]
    return float(sum(T[full[i], full[i + 1]] for i in range(len(full) - 1)))


# ------------------------------------------------------------
# 2-opt + Or-opt 混合局部搜索
# ------------------------------------------------------------
def two_opt(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    n = len(perm)
    full = [0] + list(perm) + [0]
    improved = True
    iters = 0
    while improved:
        improved = False
        for i in range(1, n):
            for k in range(i + 1, n + 1):
                a, b = full[i - 1], full[i]
                c, d = full[k], full[k + 1]
                if T[a, c] + T[b, d] < T[a, b] + T[c, d] - 1e-9:
                    full[i : k + 1] = full[i : k + 1][::-1]
                    improved = True
                    iters += 1
    return full[1:-1], iters


def or_opt(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    n = len(perm)
    full = [0] + list(perm) + [0]
    improved = True
    iters = 0
    while improved:
        improved = False
        for seg_len in (1, 2, 3):
            for i in range(1, n - seg_len + 2):
                seg = full[i : i + seg_len]
                a, b = full[i - 1], full[i + seg_len]
                cost_remove = T[a, seg[0]] + T[seg[-1], b] - T[a, b]
                base = full[:i] + full[i + seg_len :]
                for j in range(len(base) - 1):
                    if i - 1 <= j <= i:
                        continue
                    p, q = base[j], base[j + 1]
                    cost_insert = T[p, seg[0]] + T[seg[-1], q] - T[p, q]
                    if cost_insert - cost_remove < -1e-9:
                        full = base[: j + 1] + seg + base[j + 1 :]
                        improved = True
                        iters += 1
                        break
                if improved:
                    break
            if improved:
                break
    return full[1:-1], iters


def hybrid_polish(perm: list[int], T: np.ndarray) -> tuple[list[int], int]:
    total = 0
    while True:
        perm, k1 = two_opt(perm, T)
        perm, k2 = or_opt(perm, T)
        total += k1 + k2
        if k1 + k2 == 0:
            return perm, total
