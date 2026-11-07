"""把「自己的长期投注流水」接进累积审计（纯本地工具）。

仓库里的 ``audit.py`` 从 ``picks/`` 扫描本项目自己生成的存档；本模块让你把
在别处记录的真实投注流水（CSV / JSONL / JSON）用**完全相同**的统计口径做审计。

口径一致性：这里不重写任何统计量 —— 把每条记录转成 ``settlement.settle()``
的返回结构，直接交给 ``audit.summarize()`` 与 ``audit.format_audit()``。
因此命中率、Wilson 区间、单边二项 p 值、返奖率、功效阈值都与项目自身
审计逐位一致，不存在「两套算法算出两个结论」的问题。

隐私边界（重要）：
- 只读取用户显式传入的文件，不扫描目录、不联网；
- 报告只打印聚合数字；
- ``--anonymize-out`` 导出的记录仅含 bets/wins/cost/payout，且顺序固定种子打乱；
- **绝不写回** ``audit_aggregate.json``、``picks/`` 或仓库内任何被发布的数据文件。
  仓库里那份脱敏聚合数据只对应项目自身 99 份存档，不应被外部记录污染。

每行/每条记录的字段：

    mode    必填，直选 / 组选6 / 组选3
    bets    必填，该期投注注数（正整数）
    wins    必填，该期中奖注数（非负整数，且不超过 bets）
    cost    必填，该期成本，元（非负有限数）
    payout  必填，该期派奖，元（非负有限数）
    period  可选，期号，仅用于重复检查；脱敏导出时丢弃
    note    可选，备注；脱敏导出时丢弃

独立运行：

    python audit_external.py --file my_records.csv
    python audit_external.py --file my_records.jsonl --json
    python audit_external.py --file my_records.csv --anonymize-out aggregate.json
    python audit_external.py --template-out my_template.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import audit
import evaluation as EV

REQUIRED = ('mode', 'bets', 'wins', 'cost', 'payout')
DISCARD_ON_EXPORT = ('period', 'note')
ANONYMIZE_SEED = 20261107
COLUMNS = ('mode', 'bets', 'wins', 'cost', 'payout', 'period', 'note')


def _int(value, name, where):
    try:
        out = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f'{where}: {name} 必须是整数，实际 {value!r}')
    return out


def _money(value, name, where):
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f'{where}: {name} 必须是数字，实际 {value!r}')
    if not math.isfinite(out) or out < 0:
        raise ValueError(f'{where}: {name} 必须是非负有限数，实际 {value!r}')
    return out


def _validate(row, where):
    """单条记录 → 规范化 dict；任何字段不合法都抛带定位信息的 ValueError。"""
    missing = [k for k in REQUIRED if row.get(k) in (None, '')]
    if missing:
        raise ValueError(f'{where}: 缺少必填字段 {", ".join(missing)}')
    mode = str(row['mode']).strip()
    if mode not in EV.PLAY_RULES:
        raise ValueError(f'{where}: 未知玩法 {mode!r}，'
                         f'可用 {", ".join(EV.PLAY_RULES)}')
    bets = _int(row['bets'], 'bets', where)
    wins = _int(row['wins'], 'wins', where)
    if bets <= 0:
        raise ValueError(f'{where}: bets 必须为正整数，实际 {bets}')
    if wins < 0 or wins > bets:
        raise ValueError(f'{where}: wins 必须在 [0, {bets}] 内，实际 {wins}')
    out = {
        'mode': mode,
        'bets': bets,
        'wins': wins,
        'cost': _money(row['cost'], 'cost', where),
        'payout': _money(row['payout'], 'payout', where),
    }
    if row.get('period') not in (None, ''):
        out['period'] = str(row['period']).strip()
    return out


def read_rows(path):
    """按扩展名读取原始行（尚未校验）。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f'{path}: CSV 为空或缺少表头')
            return [dict(r) for r in reader], list(reader.fieldnames)
    if suffix == '.jsonl':
        with path.open('r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows, sorted({k for r in rows for k in r})
    if suffix == '.json':
        with path.open('r', encoding='utf-8') as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f'{path}: JSON 顶层必须是记录数组')
        return rows, sorted({k for r in rows for k in r})
    raise ValueError(f'不支持的格式 {suffix}（支持 .csv / .jsonl / .json）')


def load_records(path):
    """读取并逐条校验，返回规范化记录列表。"""
    rows, columns = read_rows(path)
    absent = [k for k in REQUIRED if k not in columns]
    if absent:
        raise ValueError(f'{path}: 缺少必填列 {", ".join(absent)}')
    records = [_validate(r, f'{path} 第 {i + 1} 条') for i, r in enumerate(rows)]
    if not records:
        raise ValueError(f'{path}: 没有有效记录')
    return records


def to_reports(records):
    """转成 settlement.settle() 的返回结构，以便复用 audit.summarize()。"""
    reports = []
    for r in records:
        reports.append({
            'mode': r['mode'],
            'draws_checked': 1,
            'bets': r['bets'],
            'win_bets_total': r['wins'],
            'cost_total': r['cost'],
            'return_total': r['payout'],
            # summarize 只从 details 里取 period 做去重统计
            'details': [{'period': r['period']}] if 'period' in r else [],
        })
    return reports


def analyze(records):
    """外部流水 → 与项目自身审计同口径的汇总 + 数据质量信息。"""
    summary = audit.summarize(to_reports(records))
    periods = [r['period'] for r in records if 'period' in r]
    duplicates = sorted(p for p, n in Counter(periods).items() if n > 1)
    return {
        'summary': summary,
        'records': len(records),
        'total_bets': sum(r['bets'] for r in records),
        'total_wins': sum(r['wins'] for r in records),
        'total_cost': sum(r['cost'] for r in records),
        'total_payout': sum(r['payout'] for r in records),
        'periods_provided': len(periods),
        'duplicate_periods': duplicates,
        'source': 'external',
    }


def anonymize(records, seed=ANONYMIZE_SEED):
    """脱敏聚合：只留四个数值字段，固定种子打乱期序，不含期号与备注。"""
    out = [{k: r[k] for k in ('bets', 'wins', 'cost', 'payout')} for r in records]
    random.Random(seed).shuffle(out)
    return out


def format_external_report(result):
    """可读报告：审计正文直接复用 audit.format_audit，前后补口径与隐私说明。"""
    lines = [
        '=' * 66,
        '外部投注流水审计（本地数据，与仓库发布数据无关）',
        '=' * 66,
        f"读入 {result['records']} 条逐期记录，合计 {result['total_bets']} 注、"
        f"中奖 {result['total_wins']} 注。",
        f"提供期号 {result['periods_provided']} 条；"
        f"重复期号 {len(result['duplicate_periods'])} 个"
        + (f"（{', '.join(result['duplicate_periods'][:5])}…）"
           if result['duplicate_periods'] else ''),
        '统计口径与项目自身累积审计完全相同（复用 audit.summarize / format_audit）。',
        '-' * 66,
    ]
    lines.append(audit.format_audit(
        result['summary'], title='累积审计：外部投注记录 vs 理论随机基线'))
    lines += [
        '-' * 66,
        '提醒：这只是对你自己记录的账目做统计，不改变任何结论。',
        '单注中奖概率由玩法覆盖率决定，与选号方式无关；',
        '样本量未达功效阈值前，命中率高于理论值不构成「有优势」的证据。',
        '本工具不会写回仓库内的 audit_aggregate.json 或 picks/ 目录。',
        '=' * 66,
    ]
    return '\n'.join(lines)


def write_template(path):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        # 示例行，明确不是真实投注记录
        w.writerow({'mode': '组选6', 'bets': 5, 'wins': 0, 'cost': 10,
                    'payout': 0, 'period': '2026001',
                    'note': '示例行，请替换为你自己的记录'})
        w.writerow({'mode': '组选6', 'bets': 5, 'wins': 1, 'cost': 10,
                    'payout': 173, 'period': '2026002',
                    'note': '示例行，请替换为你自己的记录'})


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='外部投注流水审计（纯本地）')
    parser.add_argument('--file', default=None, help='CSV / JSONL / JSON 记录文件')
    parser.add_argument('--json', action='store_true', help='输出结构化 JSON')
    parser.add_argument('--anonymize-out', default=None,
                        help='导出脱敏聚合记录（仅四个数值字段，顺序打乱）')
    parser.add_argument('--template-out', default=None,
                        help='生成空白记录模板 CSV 并退出')
    args = parser.parse_args()

    if args.template_out:
        write_template(args.template_out)
        print(f'模板已写入 {args.template_out}（含两行示例，请替换成真实记录）')
        return 0
    if not args.file:
        parser.error('需要 --file，或用 --template-out 生成模板')

    try:
        records = load_records(args.file)
        result = analyze(records)
        if args.anonymize_out:
            target = Path(args.anonymize_out)
            target.write_text(json.dumps(anonymize(records), ensure_ascii=False,
                                         indent=2), encoding='utf-8')
            print(f'脱敏记录已写入 {target}；'
                  '请不要把它覆盖仓库内的 audit_aggregate.json。')
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_external_report(result))
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f'读取失败：{e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
