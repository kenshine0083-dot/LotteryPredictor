"""导出脱敏聚合数据（仅本地维护用，复现时不需要运行）。

公开仓库排除了 picks/（真实投注存档含期号、时间戳、选号与个人行为记录），
但 README 里的审计结论因此无法被第三方重算。本脚本把存档对账结果压成
纯数值的逐期记录，用于生成可随仓库发布的 audit_aggregate.json。

脱敏规则：
- 不写期号、开奖号码、选号号码、存档文件名、创建时间
- 每条记录只保留 4 个数：当期注数 / 中奖注数 / 成本 / 回报
- 逐期顺序用固定种子打乱，使「第几期命中」无法对应到真实期号
- 去重期数只以聚合计数形式给出，不附任何期号列表

汇总统计（总注数、命中率、净损益、返奖率、Wilson 区间、二项检验 p 值）
全部保留，可由 reproduce.py 从这些数值重新算出来。

运行：
    python export_aggregate.py                 # 写 audit_aggregate.json
    python export_aggregate.py --check         # 只校验现有文件与本地存档是否一致
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / 'audit_aggregate.json'
SHUFFLE_SEED = 20261107
SCHEMA_VERSION = 1

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def build(picks_dir=None, excel_path=None):
    """返回脱敏聚合 dict；无可用存档时返回 None。"""
    import audit

    reports, errors = audit.collect(picks_dir, excel_path)
    summary = audit.summarize(reports)
    if not summary:
        return None

    modes = {}
    for mode, s in summary.items():
        records = []
        periods = set()
        archives = 0
        for r in reports:
            if r['mode'] != mode or r['draws_checked'] == 0:
                continue
            archives += 1
            stake = float(r['stake_per_bet'])
            bets = int(r['bets'])
            for d in r['details']:
                periods.add(d['period'])
                records.append({
                    'bets': bets,
                    'wins': int(d['win_bets']),
                    'cost': round(bets * stake, 2),
                    'payout': round(float(d['payout']), 2),
                })
        # 打乱逐期顺序：不改变任何汇总统计，但切断与真实期号的对应关系
        rng = np.random.default_rng(SHUFFLE_SEED)
        rng.shuffle(records)
        modes[mode] = {
            'archives': archives,
            'period_count': len(periods),
            'per_draw': records,
        }

    return {
        'schema_version': SCHEMA_VERSION,
        'note': ('由本地 picks/ 存档对账结果导出。不含期号、开奖号码、选号号码、'
                 '存档文件名与时间戳；逐期顺序已用固定种子打乱。'
                 '仅保留可重算汇总统计所需的数值。'),
        'source': 'picks/ + 历史数据.xlsx，经 settlement.settle 对账',
        'hash_note': 'per_draw 的条数是「存档 × 期」的对账记录数，非去重期数',
        'modes': modes,
    }


def summarize_payload(payload):
    """从脱敏数值重算汇总（与 audit.summarize 同口径，但不接触任何原始数据）。"""
    import evaluation as EV
    import audit

    out = {}
    for mode, block in payload['modes'].items():
        rows = block['per_draw']
        bets = sum(r['bets'] for r in rows)
        wins = sum(r['wins'] for r in rows)
        cost = round(sum(r['cost'] for r in rows), 2)
        ret = round(sum(r['payout'] for r in rows), 2)
        p0 = EV.PLAY_RULES[mode]['hit_per_bet']
        lo, hi = EV.wilson_interval(wins, bets) if bets else (0.0, 0.0)
        out[mode] = {
            'archives': block['archives'],
            'unique_periods': block['period_count'],
            'bets': bets,
            'wins': wins,
            'observed_rate': wins / bets if bets else 0.0,
            'theoretical_rate': p0,
            'wilson': [lo, hi],
            'p_value': audit._binom_p(wins, bets, p0),
            'cost': cost,
            'return': ret,
            'pnl': round(ret - cost, 2),
            'actual_return_rate': ret / cost if cost else 0.0,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='导出脱敏审计聚合数据')
    parser.add_argument('--dir', default=None, help='picks 存档目录')
    parser.add_argument('--excel', default=None, help='历史数据 xlsx')
    parser.add_argument('--check', action='store_true',
                        help='校验现有 json 与本地存档是否一致，不写文件')
    args = parser.parse_args()

    payload = build(args.dir, args.excel)
    if payload is None:
        print('没有可对账的 picks/ 存档。复现请直接运行 reproduce.py，'
              '它使用仓库内已发布的聚合数据。')
        return 0

    if args.check:
        if not OUT_PATH.exists():
            print('audit_aggregate.json 不存在，请先运行一次导出。')
            return 1
        published = json.loads(OUT_PATH.read_text(encoding='utf-8'))
        now = summarize_payload(payload)
        old = summarize_payload(published)
        same = all(
            set(now) == set(old)
            and all(abs(now[m][k] - old[m][k]) < 1e-9
                    for k in ('bets', 'wins', 'cost', 'return', 'pnl', 'p_value'))
            for m in now
        )
        print('聚合数据一致' if same else '聚合数据与本地存档不一致，需重新导出')
        return 0 if same else 1

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'已写入 {OUT_PATH.name}（{OUT_PATH.stat().st_size / 1024:.1f} KB）')
    for mode, s in summarize_payload(payload).items():
        print(f"  {mode}: {s['bets']} 注 / {s['wins']} 中 / "
              f"命中率 {s['observed_rate']:.6f} / p={s['p_value']:.4f} / "
              f"净损益 {s['pnl']:+.0f} 元")
    return 0


if __name__ == '__main__':
    sys.exit(main())
