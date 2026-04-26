"""
问题 1 附加：枚举所有"总运输时间 = 最优值"的等价最优路径
方法：在 Held-Karp DP 上保留所有等值前驱，反向回溯所有最优 tour
输出：
  results/基础模型/qubo_v1_q1_all_optima.json
  tables/tab_01_q1_all_optima.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_XLSX = ROOT / "参考算例.xlsx"
OUT_RESULT = ROOT / "results/基础模型"
OUT_TABLE = ROOT / "tables"

N = 15

T_mat = pd.read_excel(DATA_XLSX, sheet_name=1, header=0, index_col=0).values.astype(int)
T = T_mat[: N + 1, : N + 1]


def held_karp_all_optima(T: np.ndarray, n: int):
    INF = 10**9
    full = (1 << n) - 1
    dp = np.full((1 << n, n + 1), INF, dtype=np.int64)
    # parents[mask][j] = 所有能达到最优代价 dp[mask][j] 的前驱集合
    parents: dict[tuple[int, int], list[int]] = {}

    for j in range(1, n + 1):
        dp[1 << (j - 1), j] = T[0, j]
        parents[(1 << (j - 1), j)] = [0]  # 0 表示来自 depot

    for mask in range(1, 1 << n):
        for j in range(1, n + 1):
            if not (mask & (1 << (j - 1))):
                continue
            cur = dp[mask, j]
            if cur >= INF:
                continue
            for k in range(1, n + 1):
                if mask & (1 << (k - 1)):
                    continue
                new_mask = mask | (1 << (k - 1))
                new_cost = cur + int(T[j, k])
                key = (new_mask, k)
                if new_cost < dp[new_mask, k]:
                    dp[new_mask, k] = new_cost
                    parents[key] = [j]
                elif new_cost == dp[new_mask, k]:
                    parents.setdefault(key, []).append(j)

    # 找最优终点集合
    best_cost = INF
    end_candidates: list[int] = []
    for j in range(1, n + 1):
        c = int(dp[full, j]) + int(T[j, 0])
        if c < best_cost:
            best_cost = c
            end_candidates = [j]
        elif c == best_cost:
            end_candidates.append(j)

    # 反向回溯所有最优 tour
    all_tours: list[list[int]] = []

    def backtrack(mask: int, j: int, path: list[int]):
        if mask == 0:
            # 到达 depot，path 是从终点到起点的反序（不含 depot）
            tour = [0] + path[::-1] + [0]
            all_tours.append(tour)
            return
        key = (mask, j)
        for prev in parents.get(key, []):
            if prev == 0:
                # 直接来自 depot，进入 base case
                if mask == (1 << (j - 1)):
                    backtrack(0, 0, path + [j])
            else:
                new_mask = mask ^ (1 << (j - 1))
                backtrack(new_mask, prev, path + [j])

    for end in end_candidates:
        backtrack(full, end, [])

    # 去重（理论上 backtrack 不会产生重复，但保险）
    uniq = list({tuple(t) for t in all_tours})
    uniq.sort()
    return best_cost, [list(t) for t in uniq]


def main():
    best_cost, tours = held_karp_all_optima(T, N)
    # 同时收集每条 tour 的反向版本，看 TSP 是否有方向对称
    rev_set = {tuple(reversed(t)) for t in tours}
    tour_set = {tuple(t) for t in tours}
    is_symmetric_dataset = bool(rev_set & tour_set)  # 反向是否也在最优解集中

    print(f"问题 1 最优总运输时间 = {best_cost}")
    print(f"等价最优路径条数 = {len(tours)}")
    print(f"是否含方向对称对（反向也是最优）= {is_symmetric_dataset}")
    print(f"\n前 10 条示例：")
    for t in tours[:10]:
        print("  " + " -> ".join(map(str, t)))

    # 落盘
    result = dict(
        problem="Q1: 全部等价最优 tour 枚举",
        n_customers=N,
        best_total_travel_time=best_cost,
        n_optimal_tours=len(tours),
        contains_reverse_pair=is_symmetric_dataset,
        all_optimal_tours=tours,
    )
    out_json = OUT_RESULT / "qubo_v1_q1_all_optima.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[写出] {out_json.relative_to(ROOT)}")

    # 表格：每行一条最优 tour
    df = pd.DataFrame(
        dict(
            序号=range(1, len(tours) + 1),
            总时间=[best_cost] * len(tours),
            路径=[" -> ".join(map(str, t)) for t in tours],
        )
    )
    out_csv = OUT_TABLE / "tab_01_q1_all_optima.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[写出] {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
