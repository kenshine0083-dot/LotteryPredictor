"""开奖数据随机性检验

给模型一个前置判断：这批数据里到底有没有可被学习的结构。
如果连「独立均匀随机」都不能拒绝，那么任何预测模型的理论命中率上限
就是 k/1000（k 为投注注数），再复杂的网络也只是在拟合噪声。

九项检验覆盖三个层面：
- 边际分布：各位数字是否均匀
- 横截面依赖：三位之间是否独立
- 时间依赖：期与期之间是否独立、重复号形态与和值是否符合理论分布

结果全部 p > 0.05 即表示数据与随机假设相容。

独立运行：
    python randomness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ALPHA = 0.05


def _chi2_sf(stat, dof):
    from scipy import stats
    return float(stats.chi2.sf(stat, dof))


def analyze(data):
    """data: (N, 3) 整数数组，时间升序（旧 -> 新）。返回结构化检验结果。"""
    data = np.asarray(data, dtype=np.int64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError('data 必须是 (N, 3) 的开奖号码矩阵')
    n = len(data)
    if n < 100:
        raise ValueError(f'样本量过小（{n} 期），统计检验无意义')

    tests = []

    def add(name, stat, dof, p, note=''):
        tests.append({'name': name, 'stat': float(stat), 'dof': int(dof),
                      'p': float(p), 'note': note,
                      'pass': bool(p > ALPHA)})

    # ---- 1~3. 各位数字均匀性 ----
    for pos in range(3):
        counts = np.bincount(data[:, pos], minlength=10).astype(float)
        expected = np.full(10, n / 10.0)
        stat = float(np.sum((counts - expected) ** 2 / expected))
        add(f'第{pos + 1}位数字均匀性', stat, 9, _chi2_sf(stat, 9),
            '最大偏差 %d 次' % int(np.abs(counts - expected).max()))

    # ---- 4. 三位相互独立性 ----
    # 三维列联表完全独立检验：dof = IJK - 1 - (I-1) - (J-1) - (K-1) = 972
    # 注意不能剔除小期望格子——那样统计量与自由度会同时失真，
    # 自由度下降更快，p 值会被系统性压低到误判。
    grid = np.zeros((10, 10, 10))
    for row in data:
        grid[row[0], row[1], row[2]] += 1
    marg = [np.bincount(data[:, i], minlength=10) / n for i in range(3)]
    indep = np.einsum('i,j,k->ijk', marg[0], marg[1], marg[2]).reshape(-1)
    obs = grid.reshape(-1)
    expect = indep * n
    keep = expect > 0
    stat = float(np.sum((obs[keep] - expect[keep]) ** 2 / expect[keep]))
    dof = 972
    add('三位数字相互独立性', stat, dof, _chi2_sf(stat, dof),
        '1000 种组合已出现 %d 种' % int((obs > 0).sum()))

    # ---- 5~7. 期际独立性（上期 -> 本期转移矩阵） ----
    for pos in range(3):
        trans = np.zeros((10, 10))
        for t in range(1, n):
            trans[data[t - 1, pos], data[t, pos]] += 1
        row_sum = trans.sum(axis=1, keepdims=True)
        col_sum = trans.sum(axis=0, keepdims=True)
        expect = row_sum * col_sum / trans.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = float(np.nansum((trans - expect) ** 2 / expect))
        dof = 81
        add(f'第{pos + 1}位期际独立性', stat, dof, _chi2_sf(stat, dof))

    # ---- 8. 重复号形态 ----
    distinct_pairs = ((data[:, 0] != data[:, 1]).astype(int)
                      + (data[:, 1] != data[:, 2]).astype(int)
                      + (data[:, 0] != data[:, 2]).astype(int))
    obs3 = np.array([(distinct_pairs == 3).sum(), (distinct_pairs == 2).sum(),
                     (distinct_pairs == 0).sum()], dtype=float)
    theo3 = np.array([0.72, 0.27, 0.01]) * n
    stat = float(np.sum((obs3 - theo3) ** 2 / theo3))
    add('重复号形态(组6/组3/豹子)', stat, 2, _chi2_sf(stat, 2),
        '观测 %s' % (obs3 / n).round(4).tolist())

    # ---- 9. 和值分布 ----
    sums = data.sum(axis=1)
    theo = np.zeros(28)
    for a in range(10):
        for b in range(10):
            for c in range(10):
                theo[a + b + c] += 1 / 1000
    obs_sum = np.array([(sums == s).sum() for s in range(28)], dtype=float)
    expect_sum = theo * n
    stat = float(np.sum((obs_sum - expect_sum) ** 2 / np.maximum(expect_sum, 1e-9)))
    add('和值分布', stat, 27, _chi2_sf(stat, 27),
        '均值 %.3f（理论 13.5）' % sums.mean())

    return {'n': n, 'tests': tests, 'alpha': ALPHA,
            'all_pass': all(t['pass'] for t in tests)}


def format_report(result):
    """检验结果 → 可读文本报告。"""
    if not result:
        return '【随机性检验】无数据'
    lines = ['=' * 66,
             f"开奖数据随机性检验（{result['n']} 期）",
             '=' * 66,
             f"{'检验项':<24}{'卡方':>10}{'自由度':>8}{'p值':>9}{'结论':>8}"]
    lines.append('-' * 66)
    for t in result['tests']:
        lines.append(f"{t['name']:<24}{t['stat']:>10.2f}{t['dof']:>8d}"
                     f"{t['p']:>9.4f}{'  符合' if t['pass'] else '  异常':>8}")
        if t['note']:
            lines.append(f"{'':<24}{t['note']}")
    lines.append('-' * 66)
    failed = [t['name'] for t in result['tests'] if not t['pass']]
    if failed:
        lines.append(f"有 {len(failed)} 项检验显著（{('、'.join(failed))}）。")
        lines.append('这可能是真实结构，也可能是多重比较下的偶然结果——'
                     '注意 9 项同时检验时，即使数据完全随机，'
                     '也约有 37% 概率至少出现一项 p<0.05。')
    else:
        lines.append('全部 9 项检验 p > 0.05，数据与「独立均匀随机」假设完全相容。')
        lines.append('含义：不存在可被模型利用的结构。任何选号方式的单期中奖概率')
        lines.append('都是 k/1000（k 为互不重复的注数），与选号策略无关。')
    lines.append('=' * 66)
    return '\n'.join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    import pandas as pd
    from paths import resolve
    df = pd.read_excel(resolve('历史数据.xlsx'), engine='openpyxl')
    df = df.sort_values('Period')
    data = np.array([[int(t) for t in str(r).split()] for r in df['Numbers']])
    print(format_report(analyze(data)))


if __name__ == '__main__':
    main()
