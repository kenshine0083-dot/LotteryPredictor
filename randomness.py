"""开奖数据随机性检验

给模型一个前置判断：这批数据里到底有没有可被学习的结构。
如果连「独立均匀随机」都不能拒绝，那么任何预测模型的理论命中率上限
就是 k/1000（k 为投注注数），再复杂的网络也只是在拟合噪声。

九项检验覆盖三个层面：
- 边际分布：各位数字是否均匀
- 横截面依赖：三位之间是否独立
- 时间依赖：期与期之间是否独立、重复号形态与和值是否符合理论分布

九项检验构成同一个检验族，逐项以 α=0.05 裸判会产生累积误判：
即使数据完全随机，至少一项 p<0.05 的概率也有 1-0.95^9 ≈ 37%。
因此整体结论一律以 Holm-Bonferroni 校正后的 p 值为准（控制族错误率 FWER）。
单项原始 p 值仍保留，供诊断使用，但不作为「数据有结构」的证据。

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


def holm_adjust(p_values):
    """Holm-Bonferroni step-down 校正，返回与输入同序的校正后 p 值。

    排序后第 rank 小的 p 值乘以 (m - rank)，再取前缀最大值保证单调，
    最后截断到 1。相比 Bonferroni 更不保守，且同样把 FWER 控制在 α 以下。
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


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

    # ---- 多重比较校正：九项检验属于同一族，整体结论必须控制 FWER ----
    p_holm = holm_adjust([t['p'] for t in tests])
    for t, adj in zip(tests, p_holm):
        t['p_holm'] = float(adj)
        t['pass_holm'] = bool(adj > ALPHA)

    m = len(tests)
    n_nominal = sum(1 for t in tests if not t['pass'])
    n_holm = sum(1 for t in tests if not t['pass_holm'])
    return {
        'n': n, 'tests': tests, 'alpha': ALPHA, 'n_tests': m,
        'correction': 'holm',
        'familywise_error_raw': 1.0 - (1.0 - ALPHA) ** m,
        'n_nominal': n_nominal,       # 裸判显著项数
        'n_holm': n_holm,             # 校正后仍显著的项数
        # all_pass 为整体结论（校正后），all_pass_raw 保留逐项裸判结果供对照
        'all_pass': n_holm == 0,
        'all_pass_raw': n_nominal == 0,
    }


def format_report(result):
    """检验结果 → 可读文本报告。"""
    if not result:
        return '【随机性检验】无数据'
    n_tests = result.get('n_tests', len(result['tests']))
    lines = ['=' * 74,
             f"开奖数据随机性检验（{result['n']} 期，{n_tests} 项，Holm 校正）",
             '=' * 74,
             f"{'检验项':<24}{'卡方':>10}{'自由度':>8}{'p值':>9}{'Holm p':>10}{'结论':>8}"]
    lines.append('-' * 74)
    for t in result['tests']:
        adj = t.get('p_holm', t['p'])
        ok = t.get('pass_holm', t['pass'])
        lines.append(f"{t['name']:<24}{t['stat']:>10.2f}{t['dof']:>8d}"
                     f"{t['p']:>9.4f}{adj:>10.4f}{'  符合' if ok else '  异常':>8}")
        if t['note']:
            lines.append(f"{'':<24}{t['note']}")
    lines.append('-' * 74)
    lines.append(f"多重比较：{n_tests} 项逐项裸判至少一项误报的概率约 "
                 f"{result.get('familywise_error_raw', 0) * 100:.0f}%；"
                 f"结论列采用 Holm 校正后 p 值（FWER ≤ α={result['alpha']}）。")
    failed = [t['name'] for t in result['tests']
              if not t.get('pass_holm', t['pass'])]
    n_nominal = result.get('n_nominal', 0)
    if failed:
        lines.append(f"校正后仍有 {len(failed)} 项显著：{'、'.join(failed)}。")
        lines.append('这是拒绝「独立均匀随机」的统计证据，需进一步核查数据来源。')
    elif n_nominal:
        lines.append(f"裸判有 {n_nominal} 项 p<0.05，但 Holm 校正后均不显著——"
                     '属于多重比较下的偶然波动，不构成「数据有结构」的证据。')
        lines.append('整体结论：数据与「独立均匀随机」假设相容。')
        lines.append('含义：不存在可被模型利用的结构。任何选号方式的单期中奖概率')
        lines.append('都是 k/1000（k 为互不重复的注数），与选号策略无关。')
    else:
        lines.append(f"全部 {n_tests} 项检验原始 p 与 Holm 校正后 p 均 > "
                     f"{result['alpha']}，数据与「独立均匀随机」假设完全相容。")
        lines.append('含义：不存在可被模型利用的结构。任何选号方式的单期中奖概率')
        lines.append('都是 k/1000（k 为互不重复的注数），与选号策略无关。')
    lines.append('=' * 74)
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
