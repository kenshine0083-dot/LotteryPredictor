"""一键复现：把 README 里的每个数字都变成可验证的断言。

干净 clone 后直接运行，不需要 picks/、不需要模型权重、不需要 TensorFlow：

    python reproduce.py
    python reproduce.py --json     # 机器可读结果，供 CI 使用

它复用项目本体的代码路径（randomness.analyze / evaluation.wilson_interval /
audit._binom_p / audit.required_samples），从仓库内公开文件重算：
- config.json + DATA_SOURCES.md → 数据来源声明与 xlsx 自身的一致性
- 历史数据.xlsx           → 七千余期开奖的九项随机性检验（含 Holm 校正）
- audit_aggregate.json    → 脱敏后的逐期投注数值 → 累积审计账目

然后把重算结果与 README 声明的数字逐项比对，任何一项不符即以退出码 1 结束。
EXPECTED 中的数字是「文档断言」，不是 calculated 结果：文档改了数字而这里没同步，
复现就会失败，这正是它的用途——防止结论与数据在后续维护中悄悄漂移。

边界：本脚本验证的是「文档数字能否由公开数据经同一套代码重算出来」，
不验证统计算法本身是否正确（那是 test_core.py 与 test_pipeline.py 的职责）。

依赖：requirements-core.txt（numpy / pandas / scipy / openpyxl），不需要 TensorFlow。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXCEL = BASE_DIR / '历史数据.xlsx'
AGGREGATE = BASE_DIR / 'audit_aggregate.json'

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# ---- README 声明值（抄自「先说结论」「实测：它自己的账」两节）----
EXPECTED = {
    'draw_count': 7770,
    # 九项检验原始 p 值，按 randomness.analyze 的输出顺序
    'p_values': [0.67, 0.82, 0.15, 0.37, 0.48, 0.98, 0.12, 0.31, 0.21],
    'n_tests': 9,
    'holm_significant': 0,
    'audit': {
        '组选6': {
            'archives': 3, 'unique_periods': 95, 'bets': 2860, 'wins': 17,
            'observed_rate': 0.005944, 'theoretical_rate': 0.006000,
            'ratio': 0.99, 'p_value': 0.548,
            'cost': 5720.0, 'return': 2941.0, 'pnl': -2779.0,
            'return_rate_pct': 51.4,
        },
        '直选': {
            'archives': 2, 'unique_periods': 1, 'bets': 10, 'wins': 0,
            'observed_rate': 0.0, 'theoretical_rate': 0.001000,
            'ratio': 0.00, 'p_value': 1.000,
            'cost': 20.0, 'return': 0.0, 'pnl': -20.0,
            'return_rate_pct': 0.0,
        },
    },
    # 统计功效（直选，单边 α=0.05 / power=0.80）
    'power': {0.10: 638191, 0.50: 28602, 1.00: 8026},
}


class Recorder:
    def __init__(self):
        self.items = []

    def eq(self, name, actual, expected, tol=0.0):
        ok = abs(actual - expected) <= tol
        self.items.append({
            'name': name, 'ok': bool(ok),
            'actual': round(float(actual), 6),
            'expected': round(float(expected), 6),
        })
        return ok

    def flag(self, name, ok, actual, expected='—'):
        self.items.append({
            'name': name, 'ok': bool(ok),
            'actual': actual, 'expected': expected,
        })
        return bool(ok)

    @property
    def passed(self):
        return all(i['ok'] for i in self.items)


def load_draws():
    """从仓库自带的开奖数据读出 (N, 3) 整数矩阵，时间升序。"""
    import numpy as np
    import pandas as pd
    from paths import resolve
    df = pd.read_excel(resolve('历史数据.xlsx'), engine='openpyxl')
    df = df.sort_values('Period')
    data = np.array([[int(t) for t in str(r).split()] for r in df['Numbers']])
    return data


def check_data_source(rec):
    """核对随仓库发布的数据文件与 config.json / DATA_SOURCES.md 的声明。"""
    import pandas as pd

    cfg = json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))
    src = cfg.get('data_source', {})
    rec.flag('数据来源:config.json 含来源声明',
             bool(src.get('provider')) and bool(src.get('endpoint')),
             src.get('provider', '缺失'), '体彩官方 webapi')
    rec.flag('数据来源:声明接口与 fetch_url 同源',
             bool(src.get('endpoint'))
             and src['endpoint'] in cfg.get('fetch_url', ''),
             src.get('endpoint', '缺失'), 'fetch_url 前缀')
    doc = BASE_DIR / 'DATA_SOURCES.md'
    rec.flag('数据来源:说明文档已随仓库发布', doc.is_file(),
             '存在' if doc.is_file() else '缺失', '存在')

    df = pd.read_excel(EXCEL, engine='openpyxl')
    rec.eq('数据来源:期数', len(df), src.get('draw_count', -1))
    rec.flag('数据来源:列名固定为 Period / Numbers',
             list(df.columns) == ['Period', 'Numbers'], str(list(df.columns)))
    rec.flag('数据来源:期号唯一且范围与声明一致',
             bool(df['Period'].is_unique)
             and int(df['Period'].min()) == src.get('period_min')
             and int(df['Period'].max()) == src.get('period_max'),
             f"{int(df['Period'].min())}–{int(df['Period'].max())}")
    ok_fmt = all(
        len(parts) == 3 and all(len(t) == 1 and t.isdigit() and 0 <= int(t) <= 9
                                for t in parts)
        for parts in (str(x).split() for x in df['Numbers']))
    rec.flag('数据来源:开奖号均为空格分隔的 3 个 0-9 数字', ok_fmt,
             '全部合法' if ok_fmt else '存在异常行')


def check_game_profile(rec):
    """核对 config 声明的默认玩法规格与 randomness 的模板/默认值一致。"""
    import randomness as R

    cfg = json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))
    p = cfg.get('game_profile', {})
    rec.flag('玩法规格:config 声明完整',
             all(k in p for k in ('name', 'digits', 'base', 'n_outcomes')),
             ', '.join(sorted(p)), 'name/digits/base/n_outcomes')
    rec.eq('玩法规格:默认位数', p.get('digits', -1), R.DEFAULT_DIGITS)
    rec.eq('玩法规格:每位取值数', p.get('base', -1), R.DEFAULT_BASE)
    rec.eq('玩法规格:结果空间', p.get('n_outcomes', -1),
           R.DEFAULT_BASE ** R.DEFAULT_DIGITS)
    spec = R.PROFILES.get(p.get('name', ''), {})
    rec.flag('玩法规格:与 PROFILES 模板一致',
             spec.get('digits') == p.get('digits')
             and spec.get('base') == p.get('base'),
             f"{spec.get('digits')}/{spec.get('base')}",
             f"{p.get('digits')}/{p.get('base')}")


def check_randomness(rec):
    import randomness
    data = load_draws()
    result = randomness.analyze(data)

    rec.eq('开奖期数', len(data), EXPECTED['draw_count'])
    rec.eq('检验项数', result['n_tests'], EXPECTED['n_tests'])
    for t, want in zip(result['tests'], EXPECTED['p_values']):
        rec.eq('p值 · ' + t['name'], t['p'], want, tol=0.005)
    rec.eq('Holm 校正后显著项数', result['n_holm'], EXPECTED['holm_significant'])
    rec.flag('整体结论：与独立均匀随机相容', result['all_pass'],
             '相容' if result['all_pass'] else '拒绝', '相容')
    rec.flag('多重比较已校正', result.get('correction') == 'holm',
             result.get('correction'), 'holm')
    return result


def check_audit(rec):
    import evaluation as EV
    import audit
    from export_aggregate import summarize_payload

    if not AGGREGATE.exists():
        rec.flag('聚合数据文件存在', False, '缺失', '存在')
        return
    payload = json.loads(AGGREGATE.read_text(encoding='utf-8'))
    rec.flag('聚合数据 schema', payload.get('schema_version') == 1,
             payload.get('schema_version'), 1)
    summary = summarize_payload(payload)

    rec.flag('玩法集合一致', set(summary) == set(EXPECTED['audit']),
             sorted(summary), sorted(EXPECTED['audit']))
    for mode, want in EXPECTED['audit'].items():
        if mode not in summary:
            continue
        s = summary[mode]
        rec.eq(f'{mode} · 存档数', s['archives'], want['archives'])
        rec.eq(f'{mode} · 去重期数', s['unique_periods'], want['unique_periods'])
        rec.eq(f'{mode} · 投注注数', s['bets'], want['bets'])
        rec.eq(f'{mode} · 中奖注数', s['wins'], want['wins'])
        rec.eq(f'{mode} · 实际命中率', s['observed_rate'], want['observed_rate'], tol=5e-7)
        rec.eq(f'{mode} · 理论命中率', s['theoretical_rate'], want['theoretical_rate'], tol=5e-7)
        ratio = s['observed_rate'] / s['theoretical_rate']
        rec.eq(f'{mode} · 倍数', ratio, want['ratio'], tol=0.005)
        rec.eq(f'{mode} · 单边二项 p', s['p_value'], want['p_value'], tol=0.0005)
        rec.eq(f'{mode} · 成本', s['cost'], want['cost'])
        rec.eq(f'{mode} · 回报', s['return'], want['return'])
        rec.eq(f'{mode} · 净损益', s['pnl'], want['pnl'])
        rec.eq(f'{mode} · 返奖率 %', s['actual_return_rate'] * 100,
               want['return_rate_pct'], tol=0.05)
        # 独立复核：命中率上限与赔率规则必须自洽
        rec.eq(f'{mode} · 玩法单注概率', EV.PLAY_RULES[mode]['hit_per_bet'],
               want['theoretical_rate'], tol=1e-12)

    p0 = EV.PLAY_RULES['直选']['hit_per_bet']
    for lift, want in EXPECTED['power'].items():
        need = audit.required_samples(p0, lift)
        rec.eq(f'功效 · 直选 +{lift * 100:.0f}% 所需注数', need, want, tol=0.6)
        rec.flag(f'功效 · 直选 +{lift * 100:.0f}% 量级',
                 1e3 < need < 1e7, f'{need:,.0f} 注')


def main():
    parser = argparse.ArgumentParser(description='一键复现 README 中的统计结论')
    parser.add_argument('--json', action='store_true', help='输出 JSON 结果')
    args = parser.parse_args()

    missing = [p.name for p in (EXCEL, AGGREGATE) if not p.exists()]
    rec = Recorder()
    if missing:
        rec.flag('仓库内数据文件齐全', False, '缺失: ' + ', '.join(missing), '齐全')
    else:
        rec.flag('仓库内数据文件齐全', True, '齐全')
        try:
            check_data_source(rec)
        except Exception as e:
            rec.flag('数据来源核对可运行', False, f'{type(e).__name__}: {e}', '成功')
        try:
            check_game_profile(rec)
        except Exception as e:
            rec.flag('玩法规格核对可运行', False, f'{type(e).__name__}: {e}', '成功')
        try:
            check_randomness(rec)
        except Exception as e:
            rec.flag('随机性检验可运行', False, f'{type(e).__name__}: {e}', '成功')
        try:
            check_audit(rec)
        except Exception as e:
            rec.flag('累积审计可运行', False, f'{type(e).__name__}: {e}', '成功')

    if args.json:
        print(json.dumps({'passed': rec.passed, 'total': len(rec.items),
                          'items': rec.items}, ensure_ascii=False, indent=2))
        return 0 if rec.passed else 1

    lines = ['=' * 70, '一键复现：README 声明值 vs 仓库数据重算值', '=' * 70]
    for i in rec.items:
        mark = 'PASS' if i['ok'] else 'FAIL'
        lines.append(f"[{mark}] {i['name']:<34} 重算 {str(i['actual']):<14} "
                     f"声明 {i['expected']}")
    lines.append('-' * 70)
    n_ok = sum(1 for i in rec.items if i['ok'])
    lines.append(f"共 {len(rec.items)} 项，通过 {n_ok} 项，失败 {len(rec.items) - n_ok} 项")
    if rec.passed:
        lines.append('结论：README 中所有可核对的数字，均可由仓库内公开数据复算得到。')
    else:
        lines.append('结论：存在不一致，请检查是否修改了数据或文档而未同步另一处。')
    lines.append('说明：本脚本验证「数字可复算」，统计算法正确性由 test_core.py / test_pipeline.py 覆盖。')
    lines.append('=' * 70)
    print('\n'.join(lines))
    return 0 if rec.passed else 1


if __name__ == '__main__':
    sys.exit(main())
