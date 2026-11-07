"""排列3 历史数据随机性检验

在建模之前先回答一个前置问题：这批开奖数据里是否存在任何可被模型利用的
统计结构。如果连基本独立性都成立，那么任何预测模型的理论上限就是随机基线。

检验项：
1. 单一位数字的均匀性（卡方拟合优度）
2. 三位之间的相互独立性（3 维列联表卡方）
3. 期与期之间的独立性（上期号码 -> 本期号码的转移矩阵卡方）
4. 相邻期同位重复个数的分布（对照二项分布理论值）
5. 重复号形态比例（豹子/组3/组6，对照理论概率）
6. 和值分布（对照理论分布）
7. 遗漏值分布（对照几何分布）
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "历史数据.xlsx")


def load_data():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["Period"] = df["Period"].astype(int)
    df = df.sort_values("Period").reset_index(drop=True)
    nums = np.array([[int(t) for t in str(r).split()] for r in df["Numbers"]], dtype=np.int32)
    return df, nums


def line(title):
    print()
    print(title)
    print("-" * 62)


def main():
    df, nums = load_data()
    n = len(nums)
    print(f"数据文件: {DATA_PATH}")
    print(f"期数: {n}   期号范围: {df['Period'].iloc[0]} ~ {df['Period'].iloc[-1]}")
    print(f"时间跨度: {n} 期")

    # ---------- 1. 均匀性 ----------
    line("1. 单一位数字均匀性（期望各 10%，自由度 9）")
    for pos in range(3):
        counts = np.bincount(nums[:, pos], minlength=10)
        chi2, p = stats.chisquare(counts)
        print(f"  第{pos + 1}位  chi2={chi2:7.3f}  p={p:.4f}  "
              f"{'偏离显著' if p < 0.05 else '符合均匀'}")
        print(f"         频次: {counts.tolist()}")

    # ---------- 2. 三位独立性 ----------
    line("2. 三位数字相互独立性（3x10x10 列联表）")
    table = np.zeros((10, 10, 10))
    for row in nums:
        table[row[0], row[1], row[2]] += 1
    chi2_3d = 0.0
    dof_3d = 0
    flat = table.reshape(-1)
    joint_p = flat / n
    marg = [np.bincount(nums[:, i], minlength=10) / n for i in range(3)]
    indep_p = np.einsum("i,j,k->ijk", marg[0], marg[1], marg[2]).reshape(-1)
    mask = indep_p > 0
    expected = indep_p * n
    ok = expected >= 5
    chi2_3d = np.sum((flat[ok] - expected[ok]) ** 2 / expected[ok])
    dof_3d = int(ok.sum()) - (10 - 1) - (10 - 1) - (10 - 1) - 1
    p_3d = stats.chi2.sf(chi2_3d, dof_3d)
    print(f"  观测 1000 种组合，chi2={chi2_3d:.3f}  dof={dof_3d}  p={p_3d:.4f}")
    print(f"  {'存在显著交互' if p_3d < 0.05 else '与三位独立假设相容'}")
    nonzero = int((flat > 0).sum())
    print(f"  1000 种组合中出现过的: {nonzero} 种，未出现: {1000 - nonzero} 种")
    exp_nonzero = 1000 * (1 - (1 - 1 / 1000) ** n)
    print(f"  纯随机假设下期望出现 {exp_nonzero:.1f} 种  "
          f"(观测偏离 {nonzero - exp_nonzero:+.1f})")

    # ---------- 3. 期际独立性 ----------
    line("3. 期与期之间的独立性（第 t-1 期 -> 第 t 期，100x100 转移）")
    for pos in range(3):
        trans = np.zeros((10, 10))
        for t in range(1, n):
            trans[nums[t - 1, pos], nums[t, pos]] += 1
        chi2, p, dof, _ = stats.chi2_contingency(trans)
        print(f"  第{pos + 1}位  chi2={chi2:7.3f}  dof={dof}  p={p:.4f}  "
              f"{'存在相依' if p < 0.05 else '独立'}")

    # ---------- 4. 同位重复 ----------
    line("4. 相邻两期同位重复个数（理论：Binomial(3, 0.1)）")
    same = (nums[1:] == nums[:-1]).sum(axis=1)
    obs = np.bincount(same, minlength=4) / (n - 1)
    theo = [stats.binom.pmf(k, 3, 0.1) for k in range(4)]
    for k in range(4):
        print(f"  重复 {k} 位: 观测 {obs[k]:.4f}  理论 {theo[k]:.4f}  "
              f"差 {(obs[k] - theo[k]) * 100:+.2f}pp")
    exp_same = 3 * 0.1
    print(f"  平均重复位数: 观测 {same.mean():.4f}  理论 {exp_same:.4f}")

    # ---------- 5. 号码形态 ----------
    line("5. 重复号形态（豹子/组3/组6）")
    distinct = np.array([len(set(row)) for row in nums])
    obs = {3: (distinct == 3).mean(), 2: (distinct == 2).mean(), 1: (distinct == 1).mean()}
    theo = {3: 720 / 1000, 2: 270 / 1000, 1: 10 / 1000}
    names = {3: "组6(全不同)", 2: "组3(一对)", 1: "豹子"}
    chi2 = 0.0
    for k in (3, 2, 1):
        o = obs[k] * n
        e = theo[k] * n
        chi2 += (o - e) ** 2 / e
        print(f"  {names[k]:<12} 观测 {obs[k]:.4f}  理论 {theo[k]:.4f}  "
              f"({int(o)} / {e:.1f} 期)")
    print(f"  卡方={chi2:.3f}  dof=2  p={stats.chi2.sf(chi2, 2):.4f}")

    # ---------- 6. 和值 ----------
    line("6. 和值分布（理论为 3 个均匀数字之和）")
    sums = nums.sum(axis=1)
    theo_pmf = np.zeros(28)
    for a in range(10):
        for b in range(10):
            for c in range(10):
                theo_pmf[a + b + c] += 1 / 1000
    obs_pmf = np.array([(sums == s).mean() for s in range(28)])
    chi2 = np.sum((obs_pmf * n - theo_pmf * n) ** 2 / np.maximum(theo_pmf * n, 1e-9))
    print(f"  均值: 观测 {sums.mean():.3f}  理论 13.5")
    print(f"  标准差: 观测 {sums.std():.3f}  理论 {np.sqrt(3 * 8.25):.3f}")
    print(f"  卡方={chi2:.3f}  dof=27  p={stats.chi2.sf(chi2, 27):.4f}")

    # ---------- 7. 遗漏值 ----------
    line("7. 单个数字的遗漏间隔（理论：几何分布 p=0.1）")
    gaps_all = []
    for d in range(10):
        pos = np.where(nums.ravel() == d)[0]
        if len(pos) > 1:
            gaps_all.extend(np.diff(pos).tolist())
    gaps = np.array(gaps_all)
    print(f"  样本数 {len(gaps)}  平均间隔 观测 {gaps.mean():.3f}  理论 10.0")
    print(f"  中位数 观测 {np.median(gaps):.1f}  理论 6.6")
    max_gap = max(
        (np.diff(np.where(nums.ravel() == d)[0]).max() if (nums.ravel() == d).sum() > 1 else 0)
        for d in range(10)
    )
    print(f"  最大遗漏 观测 {max_gap}  理论最长约 {np.log(n * 3 / 10) / np.log(1 / 0.9):.0f}")

    print()
    print("=" * 62)
    print("结论提示：若上述 p 值普遍大于 0.05，说明数据与「独立均匀随机」")
    print("假设相容，任何模型的命中率上限即 k/1000（k=投注注数）。")


if __name__ == "__main__":
    main()
