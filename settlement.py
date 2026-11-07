"""投注选号存档与开奖对账。

生成选号时保存（含当时最新期号 base_period 与理论账目），数据更新后
核对 base_period 之后已开奖的期，报告真实命中与盈亏。

期号规则：只比较 Period > base_period，不假设 base_period+1 存在
（排列3 期号跨年跳变，如 25099 -> 26001）。

存档版本：v2 起显式写出 schema_version 字段；v1 是早期没有该字段的存档，
load_picks 会将其识别为 v1 并继续支持（字段集合相同，只是缺版本号）。
遇到高于当前版本的 schema_version 一律拒绝，避免用旧代码误读未来的格式。

独立运行：
    python settlement.py --file picks/picks_xxx.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import evaluation as EV

BASE_DIR = Path(__file__).resolve().parent
PICKS_DIR = BASE_DIR / 'picks'
# 当前写出的存档格式版本；v1 = 早期无版本字段的存档（仍可读取）
PICKS_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def save_picks(base_period, mode, picks, stake_per_bet=2.0, dir_path=None):
    """保存一次选号方案，返回文件路径。

    picks: [{'numbers': '056', 'score': 0.0094}, ...]
    """
    picks = list(picks)
    if not picks:
        raise ValueError('picks 为空，不保存')
    if mode not in EV.PLAY_RULES:
        raise ValueError(f'未知玩法：{mode}')
    odds = EV.play_odds(mode, len(picks), stake_per_bet)
    record = {
        'schema_version': PICKS_SCHEMA_VERSION,
        'base_period': int(base_period),
        'mode': mode,
        'stake_per_bet': stake_per_bet,
        'created': datetime.now().isoformat(timespec='seconds'),
        'picks': [p['numbers'] for p in picks],
        'scores': [round(float(p.get('score', 0.0)), 6) for p in picks],
        'theoretical': {
            'hit_rate': odds['coverage'],
            'cost': odds['cost'],
            'expected_return': odds['expected_return'],
            'return_rate': odds['return_rate'],
        },
        # 存档自带口径声明，避免日后翻看时误读 scores 的含义
        'note': (f'scores 是模型打分，只用于从全部组合中挑出这 {len(picks)} 注，'
                 f'不是中奖概率；每注真实中奖概率恒为 '
                 f'{odds["coverage"] * 100 / len(picks):.2f}%。'
                 f'本方案 {len(picks)} 注互不重复的{mode}，单期中奖概率恒为 '
                 f'{odds["coverage"] * 100:.2f}%，与随机选号完全相同。'),
    }
    out_dir = Path(dir_path) if dir_path else PICKS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = out_dir / f'picks_{base_period}_{mode}_{stamp}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def load_picks(path):
    """读取选号文件，损坏、字段缺失或版本不受支持时抛带原因的 ValueError。

    缺少 schema_version 的旧存档按 v1 处理，并在返回值中补上版本字段，
    这样调用方可以统一按 record['schema_version'] 判断，而不必到处兜底。
    """
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        raise ValueError(f'选号文件无法读取：{path}（{e}）')
    for key in ('base_period', 'mode', 'picks'):
        if key not in record:
            raise ValueError(f'选号文件缺少字段 {key}：{path}')
    version = record.get('schema_version', LEGACY_SCHEMA_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f'选号文件 schema_version 非法：{version!r}（{path}）')
    if not LEGACY_SCHEMA_VERSION <= version <= PICKS_SCHEMA_VERSION:
        raise ValueError(f'选号文件 schema_version={version} 不受支持'
                         f'（当前支持 {LEGACY_SCHEMA_VERSION}–{PICKS_SCHEMA_VERSION}）：{path}')
    record['schema_version'] = version
    if record['mode'] not in EV.PLAY_RULES:
        raise ValueError(f'选号文件玩法非法：{record["mode"]}')
    return record


def load_draws_after(excel_path, base_period, df=None):
    """返回 Period > base_period 的 (期号, 开奖串) 列表（升序）。

    不假设 base_period+1 存在（跨年期号跳变）。
    df: 可传入已读入的 Excel DataFrame（对账多份存档时只读一次文件）。
    """
    import pandas as pd
    if df is None:
        df = pd.read_excel(excel_path, engine='openpyxl')
    df = df[df['Period'] > int(base_period)].sort_values(by='Period', ascending=True)
    draws = []
    for _, row in df.iterrows():
        try:
            numbers = str(row['Numbers']).strip()
            parts = numbers.split() if ' ' in numbers else list(numbers)
            if len(parts) == 3 and all(p.isdigit() and 0 <= int(p) <= 9 for p in parts):
                draws.append((int(row['Period']), numbers))
        except (ValueError, TypeError):
            continue
    return draws


def settle(picks_path, excel_path=None, stake_per_bet=None, df=None):
    """核对一份选号文件在已开奖期上的真实命中。

    df: 预读的 Excel DataFrame（GUI 批量对账时复用，避免重复读文件）。
    返回 dict：draws_checked / hit_draws / win_bets_total / return_total /
    cost_total / pnl / longest_lose_streak / details。
    """
    record = load_picks(picks_path)
    if excel_path is None:
        from paths import resolve
        excel_path = resolve('历史数据.xlsx')
    mode = record['mode']
    bets = record['picks']
    stake = float(record.get('stake_per_bet', 2.0) if stake_per_bet is None
                  else stake_per_bet)
    draws = load_draws_after(excel_path, record['base_period'], df=df)
    rule = EV.PLAY_RULES[mode]
    win_flags = []
    details = []
    return_total = 0.0
    cost_total = 0.0
    skipped = 0
    for period, actual in draws:
        try:
            win = sum(1 for p in bets if EV.bet_hits(p, actual, mode))
        except ValueError:
            skipped += 1
            continue
        cost_total += len(bets) * stake
        payout = win * rule['payout']
        return_total += payout
        win_flags.append(win > 0)
        details.append({'period': period, 'actual': actual,
                        'win_bets': win, 'payout': payout})
    n = len(win_flags)
    hit_draws = sum(win_flags)
    return {
        'file': str(picks_path),
        'mode': mode,
        'base_period': record['base_period'],
        'bets': len(bets),
        'draws_checked': n,
        'skipped_draws': skipped,
        'hit_draws': hit_draws,
        'hit_rate': hit_draws / n if n else 0.0,
        'theoretical_rate': len(bets) * rule['hit_per_bet'],
        'win_bets_total': sum(d['win_bets'] for d in details),
        'return_total': return_total,
        'cost_total': cost_total,
        'pnl': return_total - cost_total,
        'longest_lose_streak': EV.longest_false_streak(win_flags),
        'details': details,
        'stake_per_bet': stake,
        'payout': rule['payout'],
    }


def format_settlement(report):
    """对账结果 → 文本报告。"""
    if report['draws_checked'] == 0:
        return (f"选号方案 {Path(report['file']).name}：生成后暂无新开奖，"
                "请先「更新数据」后再对账。")
    lines = [
        f"对账 | {report['mode']} × {report['bets']} 注（base_period {report['base_period']} 之后）",
        f"已核对 {report['draws_checked']} 期 | 命中 {report['hit_draws']} 期"
        f" | 实际命中率 {report['hit_rate']:.4f} vs 理论 {report['theoretical_rate']:.4f}",
        f"中奖注数合计 {report['win_bets_total']} | 回报 {report['return_total']:.0f} 元"
        f" | 成本 {report['cost_total']:.0f} 元 | 净损益 {report['pnl']:+.0f} 元",
        f"最长连败 {report['longest_lose_streak']} 期 | 单注奖金 {report['payout']:.0f} 元，"
        f"返奖率约 52%，长期期望为负",
    ]
    for d in report['details'][-10:]:
        mark = '中' if d['win_bets'] else '未中'
        lines.append(f"  期 {d['period']} 开奖 {d['actual']} → {mark}"
                     + (f"（{d['win_bets']} 注，{d['payout']:.0f} 元）" if d['win_bets'] else ''))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='排列3投注对账')
    parser.add_argument('--file', required=True, help='picks json 文件')
    parser.add_argument('--excel', default=None, help='历史数据 xlsx（默认 config 指定）')
    args = parser.parse_args()
    report = settle(args.file, args.excel)
    print(format_settlement(report))


if __name__ == '__main__':
    main()
