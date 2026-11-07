"""跨存档累积审计：证明「模型选号是否真的优于随机」

单份 picks 存档只覆盖几期开奖，样本量根本不足以判断优劣。本模块把所有
历史存档累计起来，按注级口径做统计推断：

- 总投注注数 T、总中奖注数 W、实际命中率 W/T
- 理论命中率（各玩法单注概率固定：直选 1/1000、组选3 3/1000、组选6 6/1000）
- Wilson 95% 置信区间与单边二项检验（H1：优于随机）
- 实际返奖率 vs 官方 52%

同时给出统计功效分析：要检测出某一幅度的真实优势，需要多少注样本。
这个数字往往比结论本身更有说服力。

独立运行：
    python audit.py
    python audit.py --dir picks --excel 历史数据.xlsx
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import evaluation as EV
import settlement

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PICKS_DIR = BASE_DIR / 'picks'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def required_samples(p0, rel_lift, alpha=0.05, power=0.80):
    """检测 p0 相对提升 rel_lift 所需的「注」数（单边检验，正态近似）。

    p0: 基线单注命中概率；rel_lift: 目标相对提升（0.1 = 快 10%）。
    返回值量级通常远超实际可积累的注数，这本身就是结论。
    """
    p1 = p0 * (1 + rel_lift)
    if not 0 < p1 < 1:
        return float('inf')
    z_a = _z(1 - alpha)
    z_b = _z(power)
    num = (z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))) ** 2
    return num / (p1 - p0) ** 2


def _z(p):
    """标准正态分位数（Acklam 有理逼近，误差 < 1.15e-9）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0 or p >= 1:
        raise ValueError('概率必须在 (0,1) 内')
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def collect(picks_dir=None, excel_path=None):
    """扫描存档目录，逐份对账并返回报告列表（按玩法分组汇总在 summarize）。"""
    import pandas as pd

    picks_dir = Path(picks_dir) if picks_dir else DEFAULT_PICKS_DIR
    files = sorted(picks_dir.glob('picks_*.json'))
    if not files:
        return [], []

    if excel_path is None:
        from paths import resolve
        excel_path = resolve('历史数据.xlsx')
    df = pd.read_excel(excel_path, engine='openpyxl')

    reports, errors = [], []
    for f in files:
        try:
            reports.append(settlement.settle(str(f), excel_path=excel_path, df=df))
        except Exception as e:
            errors.append(f'{f.name}: {e}')
    return reports, errors


def summarize(reports):
    """按玩法汇总注级统计，并做累积显著性检验。"""
    by_mode = {}
    for r in reports:
        if r['draws_checked'] == 0:
            continue
        m = r['mode']
        acc = by_mode.setdefault(m, {'archives': 0, 'draws': 0, 'bets': 0,
                                     'wins': 0, 'cost': 0.0, 'return': 0.0,
                                     'periods': set()})
        acc['archives'] += 1
        acc['draws'] += r['draws_checked']
        # 注级口径：每期投 len(bets) 注，各注独立
        acc['bets'] += r['draws_checked'] * r['bets']
        acc['wins'] += r['win_bets_total']
        acc['cost'] += r['cost_total']
        acc['return'] += r['return_total']
        for d in r['details']:
            acc['periods'].add(d['period'])

    out = {}
    for mode, acc in by_mode.items():
        p0 = EV.PLAY_RULES[mode]['hit_per_bet']
        n, w = acc['bets'], acc['wins']
        out[mode] = {
            'archives': acc['archives'],
            'draws': acc['draws'],
            'unique_periods': len(acc['periods']),
            'bets': n,
            'wins': w,
            'observed_rate': w / n if n else 0.0,
            'theoretical_rate': p0,
            'wilson': EV.wilson_interval(w, n) if n else (0.0, 0.0),
            'p_value': _binom_p(w, n, p0),
            'cost': acc['cost'],
            'return': acc['return'],
            'pnl': acc['return'] - acc['cost'],
            'actual_return_rate': acc['return'] / acc['cost'] if acc['cost'] else 0.0,
        }
    return out


def _binom_p(w, n, p0):
    """单边二项检验 P(X >= w | n, p0)。"""
    if n == 0:
        return 1.0
    try:
        from scipy import stats
        return float(stats.binomtest(w, n, p0, alternative='greater').pvalue)
    except Exception:
        # 无 scipy 时用正态近似（样本量大时足够）
        mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
        if sd == 0:
            return 1.0
        from math import erf, sqrt
        return 0.5 * (1 - erf((w - 0.5 - mu) / (sd * sqrt(2))))


def format_audit(summary, errors=None, picks_total=None):
    """汇总 → 可读报告。"""
    if not summary:
        head = ('【累积审计】暂无可核对的历史存档（生成选号后需先「更新数据」'
                '让新开奖入库，才能对账）')
        return head

    lines = ['=' * 66, '累积审计：模型选号 vs 理论随机基线', '=' * 66]
    for mode, s in summary.items():
        lo, hi = s['wilson']
        verdict = ('显著优于随机' if s['p_value'] < 0.05 else '与随机无显著差异')
        lines.append(
            f"\n【{mode}】{s['archives']} 份存档 / {s['unique_periods']} 个不同期号"
            f"\n  投注 {s['bets']} 注，中奖 {s['wins']} 注"
            f"\n  实际命中率 {s['observed_rate']:.6f}  vs  理论 {s['theoretical_rate']:.6f}"
            f"  （倍数 {s['observed_rate'] / s['theoretical_rate']:.2f}x）"
            f"\n  Wilson 95% 区间 [{lo:.6f}, {hi:.6f}]"
            f"\n  单边二项检验 p = {s['p_value']:.4f}  → {verdict}"
            f"\n  成本 {s['cost']:.0f} 元，回报 {s['return']:.0f} 元，"
            f"净损益 {s['pnl']:+.0f} 元（实际返奖率 {s['actual_return_rate'] * 100:.1f}%）"
        )

    lines.append('\n' + '-' * 66)
    lines.append('统计功效：要证明模型真有优势，需要多少样本？')
    for lift in (0.10, 0.50, 1.00):
        n_need = required_samples(EV.PLAY_RULES['直选']['hit_per_bet'], lift)
        periods = n_need / 5  # 按每期 5 注估算
        lines.append(f"  直选命中率提升 {lift * 100:.0f}%：需约 {n_need:,.0f} 注"
                     f"（每期 5 注则约 {periods:,.0f} 期 ≈ {periods / 365:.0f} 年）")
    lines.append('  → 样本量远小于此值时，任何「命中率高于理论」都只是随机波动。')
    lines.append('=' * 66)
    if errors:
        lines.append(f"（{len(errors)} 份存档读取失败，已跳过）")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='跨存档累积审计')
    parser.add_argument('--dir', default=None, help='picks 存档目录')
    parser.add_argument('--excel', default=None, help='历史数据 xlsx')
    args = parser.parse_args()
    reports, errors = collect(args.dir, args.excel)
    print(format_audit(summarize(reports), errors, len(reports)))
    for e in errors:
        print('  错误：' + e)


if __name__ == '__main__':
    main()
