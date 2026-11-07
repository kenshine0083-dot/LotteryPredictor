"""统计核心测试：零 TensorFlow / 零 tkinter / 零显示环境，供 CI 使用。

    python test_core.py
    python test_core.py --json        # 机器可读结果

设计动机：test_pipeline.py 是 138 项全链路测试，顶层 import tkinter，
main() 一开始就要 import main / lottery_model（连带 TF 2.13、数百 MB、
Python 锁死在 3.8–3.11）。把「统计结论」交给 CI 持续验证，不能拖这套东西。

本文件只覆盖纯统计链路：
    paths / evaluation / randomness / settlement / audit /
    export_aggregate / reproduce / audit_external / combos
并额外守护一条架构约束：这些模块的顶层 import 不得引入重依赖。
若有人日后在核心模块里顺手加了 `import tensorflow`，CI 会直接变红。
另含玩法参数化、外部流水审计与项目自检钩子的回归测试。

全量 GUI / 模型测试仍由本地的 test_pipeline.py 负责，两者不互相替代。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np

RESULTS = []
QUIET = False          # --json 时静默，保证 stdout 只有一段合法的 JSON

# 核心模块：只允许顶层 import 标准库与下面四个轻量第三方包
CORE_MODULES = ('paths.py', 'evaluation.py', 'randomness.py', 'settlement.py',
                'audit.py', 'export_aggregate.py', 'reproduce.py',
                'audit_external.py', 'combos.py')
ALLOWED_TOP_LEVEL = {
    'numpy', 'pandas', 'scipy', 'openpyxl',
    '__future__', 'abc', 'argparse', 'ast', 'collections', 'csv', 'datetime',
    'itertools', 'json', 'math', 'os', 'pathlib', 'random', 're', 'shutil',
    'statistics', 'subprocess', 'sys', 'tempfile', 'time', 'typing', 'warnings',
}
# 核心模块彼此引用属于分层内依赖，同样允许；取 CORE_MODULES 的文件名（不含 .py）
ALLOWED_TOP_LEVEL |= {Path(m).stem for m in CORE_MODULES} | {'paths'}
FORBIDDEN_LOADED = ('tensorflow', 'keras', 'sklearn', 'tkinter',
                    'lottery_model', 'simulation', 'data_processor', 'gui', 'main')


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    if not QUIET:
        print(('PASS' if cond else 'FAIL'), '|', name, ('| ' + str(detail) if detail else ''))
    return bool(cond)


def test_holm_math():
    import randomness as R

    p = [0.01, 0.04, 0.03, 0.9, 0.8]
    adj = R.holm_adjust(p)
    check('Holm:已知输入的 step-down 结果',
          np.allclose(adj, [0.05, 0.12, 0.12, 1.0, 1.0]),
          str([round(x, 4) for x in adj]))
    check('Holm:校正后 p 不小于原始 p', np.all(adj >= np.array(p)))
    check('Holm:校正后 p 单调不减且不超过 1',
          np.all(np.diff(np.sort(adj)) >= 0) and np.all(adj <= 1.0))
    check('Holm:空输入 / 单项 / 全不显著',
          len(R.holm_adjust([])) == 0
          and np.allclose(R.holm_adjust([0.4]), [0.4])
          and np.allclose(R.holm_adjust([1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]))
    check('Holm:强信号不被校正抹掉',
          R.holm_adjust([1e-8, 0.9, 0.9])[0] < 0.05)

    # 九项裸判的族错误率，理论值 1 - 0.95^9
    fwer = 1.0 - 0.95 ** 9
    check('多重比较:九项裸判理论误报率约 37%',
          abs(fwer - 0.36975) < 0.001, f'{fwer:.5f}')


def test_randomness():
    import randomness as R
    import reproduce

    data = reproduce.load_draws()
    res = R.analyze(data)
    check('真实数据:期数正确', len(data) == 7770, str(len(data)))
    check('真实数据:九项检验', res['n_tests'] == 9 and len(res['tests']) == 9)
    check('真实数据:correction=holm', res.get('correction') == 'holm')
    check('真实数据:fwer 字段与理论一致',
          abs(res['familywise_error_raw'] - (1 - 0.95 ** 9)) < 1e-9)
    check('真实数据:每项 p_holm >= p',
          all(t['p_holm'] >= t['p'] - 1e-12 for t in res['tests']))
    check('真实数据:Holm 校正后显著项数为 0',
          res['n_holm'] == 0 and res['all_pass'] is True)
    check('真实数据:all_pass_raw 与 all_pass 分离',
          'all_pass_raw' in res and isinstance(res['all_pass_raw'], bool))

    report = R.format_report(res)
    check('报告:含 Holm p 列与误报率说明',
          'Holm p' in report and '37%' in report)
    check('报告:结论给出 k/1000 上限', 'k/1000' in report)

    for bad, label in ((np.zeros((10, 3), dtype=int), '样本过小'),
                       (np.zeros((200, 2), dtype=int), '形状非 (N,3)')):
        try:
            R.analyze(bad)
            ok = False
        except ValueError:
            ok = True
        check(f'analyze:拒绝{label}', ok)

    # 固定种子回归：Holm 必须不比裸判更激进，且不得抹掉真实结构
    total_nominal = total_holm = 0
    monotone = True
    for seed in range(1, 13):
        d = np.random.default_rng(seed).integers(0, 10, size=(4000, 3))
        r = R.analyze(d)
        total_nominal += r['n_nominal']
        total_holm += r['n_holm']
        monotone &= r['n_holm'] <= r['n_nominal']
    check('纯随机:12 组固定种子下 Holm 误报不多于裸判', monotone,
          f'裸判 {total_nominal} 项 / Holm {total_holm} 项')

    skewed = np.random.default_rng(5).integers(0, 10, size=(4000, 3))
    skewed[:, 0] = np.random.default_rng(6).choice(
        10, size=4000, p=[0.35] + [0.65 / 9] * 9)
    rs = R.analyze(skewed)
    check('人为偏斜:校正后仍被检出', rs['n_holm'] >= 1,
          f"Holm 显著 {rs['n_holm']} 项")


def test_randomness_generic():
    """检验不再写死 3 位 / 0-9：换玩法时理论分布与自由度必须跟着变。"""
    import randomness as R

    rng = np.random.default_rng(20261107)
    d3 = rng.integers(0, 10, size=(4000, 3))
    res3 = R.analyze(d3)
    check('通用化:默认自由度与旧版一致',
          [t['dof'] for t in res3['tests']] == [9, 9, 9, 972, 81, 81, 81, 2, 27],
          str([t['dof'] for t in res3['tests']]))
    check('通用化:默认结果空间 1000',
          res3['n_outcomes'] == 1000 and res3['digits'] == 3 and res3['base'] == 10)

    prof, total = R.outcome_profile(3, 10)
    check('通用化:形态计数 720/270/10',
          [c for _, c in prof] == [720, 270, 10] and total == 1000)
    theo = R.sum_distribution(3, 10)
    check('通用化:和值分布长度 28 且概率和为 1',
          len(theo) == 28 and abs(theo.sum() - 1.0) < 1e-12)

    d2 = rng.integers(0, 6, size=(4000, 2))
    res2 = R.analyze(d2, digits=2, base=6)
    check('通用化:(2,6) 检验项数 7 且结果空间 36',
          res2['n_tests'] == 7 and res2['n_outcomes'] == 36,
          f"n_tests={res2['n_tests']}")
    check('通用化:(2,6) 自由度正确',
          [t['dof'] for t in res2['tests']] == [5, 5, 25, 25, 25, 1, 10],
          str([t['dof'] for t in res2['tests']]))
    check('通用化:(2,6) 纯随机数据整体相容', res2['all_pass'])
    check('通用化:(2,6) 报告给出 k/36', 'k/36' in R.format_report(res2))

    biased = d2.copy()
    biased[:, 0] = rng.choice(6, size=4000, p=[0.4] + [0.6 / 5] * 5)
    check('通用化:(2,6) 人为偏斜校正后仍显著',
          R.analyze(biased, digits=2, base=6)['n_holm'] >= 1)

    d1 = rng.integers(0, 10, size=(1000, 1))
    res1 = R.analyze(d1, digits=1, base=10)
    check('通用化:(1,10) 退化为 3 项且自由度均正',
          res1['n_tests'] == 3 and all(t['dof'] > 0 for t in res1['tests']))

    def rejects(fn):
        try:
            fn()
            return False
        except ValueError:
            return True

    check('通用化:拒绝越界数字',
          rejects(lambda: R.analyze(np.full((200, 3), 10))))
    check('通用化:拒绝非法规格与列数不符',
          rejects(lambda: R.analyze(d3, digits=0, base=10))
          and rejects(lambda: R.analyze(d3, digits=3, base=1))
          and rejects(lambda: R.analyze(d3, digits=2, base=10)))
    check('通用化:拒绝过大的结果空间',
          rejects(lambda: R.analyze(d3, digits=9, base=10)))


def test_evaluation():
    import evaluation as EV

    check('玩法规则:单注概率与最大注数',
          EV.PLAY_RULES['直选']['hit_per_bet'] == 1 / 1000
          and EV.PLAY_RULES['组选3']['max_bets'] == 90
          and EV.PLAY_RULES['组选6']['max_bets'] == 120)

    for mode, rule in EV.PLAY_RULES.items():
        odds = EV.play_odds(mode, 10)
        check(f'{mode}:覆盖率 = 10 × 单注概率',
              abs(odds['coverage'] - 10 * rule['hit_per_bet']) < 1e-12)
        check(f'{mode}:成本与期望回报自洽',
              odds['cost'] == 20.0
              and abs(odds['expected_return'] - odds['coverage'] * rule['payout']) < 1e-9
              and abs(odds['expected_profit'] - (odds['expected_return'] - odds['cost'])) < 1e-9)
        check(f'{mode}:返奖率约 52%',
              abs(odds['return_rate'] - 0.52) < 0.002,
              f"{odds['return_rate'] * 100:.1f}%")
    try:
        EV.play_odds('直选', 1001)
        ok = False
    except ValueError:
        ok = True
    check('play_odds:拒绝超过 max_bets 的注数', ok)

    check('bet_hits:直选顺序敏感',
          EV.bet_hits('123', '123', '直选') and not EV.bet_hits('123', '321', '直选'))
    check('bet_hits:组选6 乱序命中 / 豹子不适用',
          EV.bet_hits('123', '3 2 1', '组选6')
          and not EV.bet_hits('111', '111', '组选6'))
    check('bet_hits:组选3 恰一对 / 全不同不适用',
          EV.bet_hits('112', '211', '组选3')
          and not EV.bet_hits('123', '123', '组选3'))
    check('canonical:统一三种输入', EV.canonical('9 4 4') == EV.canonical('944') == '449')

    lo, hi = EV.wilson_interval(17, 2860)
    check('Wilson 区间:包含观测比例且落在 [0,1]',
          0 <= lo <= 17 / 2860 <= hi <= 1, f'[{lo:.6f}, {hi:.6f}]')
    check('Wilson 区间:n=0 返回 (0,0)', EV.wilson_interval(0, 0) == (0.0, 0.0))
    check('longest_false_streak:边界',
          EV.longest_false_streak([False] * 5) == 5
          and EV.longest_false_streak([True] * 5) == 0
          and EV.longest_false_streak([False, True, False, False]) == 2)


def test_audit():
    import audit
    import evaluation as EV

    p0 = EV.PLAY_RULES['直选']['hit_per_bet']
    needs = [audit.required_samples(p0, lift) for lift in (0.1, 0.5, 1.0, 2.0)]
    check('功效:相对提升越大所需样本越少',
          all(a > b for a, b in zip(needs, needs[1:])),
          f'{[round(x) for x in needs]}')
    check('功效:量级合理且 +10% 需数十万注',
          1e5 < needs[0] < 1e6 and 1e3 < needs[-1] < 1e5)
    check('功效:无效参数返回 inf',
          audit.required_samples(p0, -1) == float('inf')
          and audit.required_samples(0.9, 1.0) == float('inf'))
    check('二项检验:w=0 时 p=1', abs(audit._binom_p(0, 2860, 0.006) - 1.0) < 1e-12)
    check('二项检验:w=n 时 p 极小', audit._binom_p(2860, 2860, 0.006) < 1e-12)
    check('二项检验:n=0 时 p=1', audit._binom_p(0, 0, 0.006) == 1.0)


def test_settlement():
    import settlement
    import evaluation as EV
    tmp = tempfile.mkdtemp()
    try:
        picks = [{'numbers': '056', 'score': 0.0094},
                 {'numbers': '123', 'score': 0.0081}]
        path = settlement.save_picks(26295, '组选6', picks, dir_path=tmp)
        rec = settlement.load_picks(path)
        check('存档:往返一致',
              rec['base_period'] == 26295 and rec['picks'] == ['056', '123'])
        check('存档:理论命中率与 play_odds 一致',
              abs(rec['theoretical']['hit_rate'] - EV.play_odds('组选6', 2)['coverage']) < 1e-12)
        check('存档:note 声明打分不是中奖概率',
              '不是中奖概率' in rec['note'] and '与随机选号完全相同' in rec['note'])
        check('存档:schema_version=2',
              rec.get('schema_version') == settlement.PICKS_SCHEMA_VERSION == 2)
        legacy = dict(rec)
        legacy.pop('schema_version', None)
        lp = Path(tmp) / 'legacy.json'
        lp.write_text(json.dumps(legacy, ensure_ascii=False), encoding='utf-8')
        check('存档:无版本字段的旧文件按 v1 兼容',
              settlement.load_picks(lp)['schema_version'] == 1)
        future = dict(rec)
        future['schema_version'] = 99
        fp = Path(tmp) / 'future.json'
        fp.write_text(json.dumps(future, ensure_ascii=False), encoding='utf-8')
        try:
            settlement.load_picks(fp)
            future_rejected = False
        except ValueError:
            future_rejected = True
        check('存档:未来版本被拒绝', future_rejected)
        malformed = dict(rec)
        malformed['schema_version'] = '2'
        mp = Path(tmp) / 'malformed.json'
        mp.write_text(json.dumps(malformed, ensure_ascii=False), encoding='utf-8')
        try:
            settlement.load_picks(mp)
            type_rejected = False
        except ValueError:
            type_rejected = True
        check('存档:版本字段非整数被拒绝', type_rejected)
        bad = Path(tmp) / 'broken.json'
        bad.write_text('{ not json', encoding='utf-8')
        try:
            settlement.load_picks(bad)
            ok = False
        except ValueError:
            ok = True
        check('存档:损坏文件被拒绝', ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_data_source():
    import reproduce
    rec = reproduce.Recorder()
    reproduce.check_data_source(rec)
    check('数据来源:声明与数据文件逐项一致',
          rec.passed and len(rec.items) >= 7,
          f'{sum(1 for i in rec.items if i["ok"])}/{len(rec.items)}')


def test_aggregate_and_reproduce():
    import audit
    import reproduce
    from export_aggregate import summarize_payload

    payload = json.loads((BASE / 'audit_aggregate.json').read_text(encoding='utf-8'))
    check('聚合数据:schema_version=1', payload.get('schema_version') == 1)
    keys = set()
    for block in payload['modes'].values():
        for row in block['per_draw']:
            keys |= set(row)
    check('聚合数据:逐期记录仅含四个数值字段',
          keys == {'bets', 'wins', 'cost', 'payout'}, str(sorted(keys)))
    text = json.dumps(payload, ensure_ascii=False)
    leaked = (re.findall(r'picks_\d+', text) + re.findall(r'20\d\d-\d\d-\d\d', text)
              + re.findall(r'\b\d{5}\b', text))
    check('聚合数据:无期号/时间戳/文件名泄漏', not leaked, str(leaked[:3]))

    summary = summarize_payload(payload)
    want = reproduce.EXPECTED['audit']['组选6']
    s = summary['组选6']
    check('聚合数据:可重算出 README 的组选6 账目',
          s['bets'] == want['bets'] and s['wins'] == want['wins']
          and abs(s['pnl'] - want['pnl']) < 1e-9
          and abs(s['p_value'] - want['p_value']) < 0.0005)
    check('聚合数据:命中率与理论值比值为 0.99 倍量级',
          abs(s['observed_rate'] / s['theoretical_rate'] - 0.99) < 0.005)

    rec = reproduce.Recorder()
    reproduce.check_randomness(rec)
    reproduce.check_audit(rec)
    check('一键复现:全部断言通过', rec.passed and len(rec.items) >= 45,
          f'{sum(1 for i in rec.items if i["ok"])}/{len(rec.items)}')

    # 文档漂移必须能被检出（改一个声明值，复现立刻失败）
    original = reproduce.EXPECTED['audit']['组选6']['bets']
    reproduce.EXPECTED['audit']['组选6']['bets'] = original + 1
    drift = reproduce.Recorder()
    reproduce.check_audit(drift)
    reproduce.EXPECTED['audit']['组选6']['bets'] = original
    check('一键复现:文档数字漂移会被检出', not drift.passed)

    # 审计格式化不依赖 picks/ 目录（干净 clone 也能出报告）
    report = audit.format_audit(summary)
    check('审计:报告可由聚合数据独立生成', '组选6' in report and len(report) > 100)


def test_dependency_boundary():
    """架构约束：核心模块顶层不得引入重依赖。"""
    offenders = []
    for name in CORE_MODULES:
        tree = ast.parse((BASE / name).read_text(encoding='utf-8'))
        for node in tree.body:            # 只看模块顶层，函数内 lazy import 不算
            if isinstance(node, ast.Import):
                roots = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split('.')[0]]
            else:
                continue
            for root in roots:
                if root not in ALLOWED_TOP_LEVEL:
                    offenders.append(f'{name}: {root}')
    check('依赖分层:核心模块顶层无重依赖', not offenders, str(offenders[:4]))

    loaded = [m for m in sys.modules
              if m.split('.')[0] in FORBIDDEN_LOADED]
    check('依赖分层:运行时未加载 TF/tkinter/模型模块', not loaded, str(loaded[:4]))


def test_new_features():
    """玩法参数化、外部流水审计、自检钩子与发布资产。"""
    import argparse
    import csv
    import audit
    import combos
    import evaluation as EV
    import randomness as R
    import audit_external as AE
    import pre_commit_checks as PC
    import reproduce

    # ---- 玩法模板与 CLI 规格 ----
    check('玩法模板:数量与默认规格',
          len(R.PROFILES) >= 5
          and R.PROFILES['arrange3']['digits'] == R.DEFAULT_DIGITS
          and R.PROFILES['arrange3']['base'] == R.DEFAULT_BASE)
    check('玩法模板:结果空间自洽且在上限内',
          all(1 < p['base'] ** p['digits'] <= R.MAX_OUTCOMES
              for p in R.PROFILES.values())
          and R.PROFILES['d2-b6']['base'] ** R.PROFILES['d2-b6']['digits'] == 36)
    ns = argparse.Namespace(profile='d2-b6', digits=None, base=None)
    check('CLI:模板给出 (2,6)', R.resolve_spec(ns) == (2, 6))
    ns2 = argparse.Namespace(profile='d2-b6', digits=4, base=None)
    check('CLI:显式参数覆盖模板', R.resolve_spec(ns2) == (4, 6))
    try:
        R.resolve_spec(argparse.Namespace(profile='nope', digits=None, base=None))
        rejected = False
    except ValueError:
        rejected = True
    check('CLI:未知模板被拒绝', rejected)

    rng = np.random.default_rng(20261107)
    sample = rng.integers(0, 6, size=(2000, 2))
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        csv_path = tmpd / 'draws.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Period', 'Numbers'])
            for i, row in enumerate(sample):
                w.writerow([i, ' '.join(map(str, row))])
        got = R.load_draws(csv_path, digits=2, base=6)
        check('CLI:CSV 载入形状与数值一致',
              got.shape == (2000, 2) and np.array_equal(got, sample))
        check('CLI:换玩法后结果空间 36',
              R.analyze(got, digits=2, base=6)['n_outcomes'] == 36)
        npy_path = tmpd / 'draws.npy'
        np.save(npy_path, sample)
        check('CLI:NPY 载入一致',
              np.array_equal(R.load_draws(npy_path, digits=2, base=6), sample))
        bad = tmpd / 'bad.csv'
        bad.write_text('Numbers\n6 1\n', encoding='utf-8')
        try:
            R.load_draws(bad, digits=2, base=6)
            out_rejected = False
        except ValueError:
            out_rejected = True
        check('CLI:越界数字被拒绝', out_rejected)
        try:
            R.load_draws(tmpd / 'x.txt', digits=2, base=6)
            suffix_rejected = False
        except ValueError:
            suffix_rejected = True
        check('CLI:不支持的格式被拒绝', suffix_rejected)

    # ---- combos 纯计算层 ----
    counts = [len(list(combos.iter_combos(m)))
              for m in ('直选', '组选6', '组选3')]
    check('combos:组合数 1000/120/90', counts == [1000, 120, 90], str(counts))
    check('combos:单注概率与玩法规则一致',
          all(abs(combos.COMBO_PER_BET[m] / 1000
                  - EV.PLAY_RULES[m]['hit_per_bet']) < 1e-12
              for m in combos.COMBO_PER_BET))

    # ---- 外部投注流水审计 ----
    records = [
        {'mode': '组选6', 'bets': 5, 'wins': 0, 'cost': 10.0, 'payout': 0.0,
         'period': '26001'},
        {'mode': '组选6', 'bets': 5, 'wins': 1, 'cost': 10.0, 'payout': 173.0,
         'period': '26002'},
        {'mode': '直选', 'bets': 3, 'wins': 0, 'cost': 6.0, 'payout': 0.0,
         'period': '26001'},
    ]
    result = AE.analyze(records)
    check('外部审计:总注数与中奖数',
          result['total_bets'] == 13 and result['total_wins'] == 1)
    check('外部审计:重复期号被报告',
          result['duplicate_periods'] == ['26001'])
    s6 = result['summary']['组选6']
    check('外部审计:p 值与项目口径一致',
          abs(s6['p_value'] - audit._binom_p(1, 10, 0.006)) < 1e-12)
    check('外部审计:净损益按记录累加',
          abs(s6['pnl'] - 153.0) < 1e-9)
    check('外部审计:脱敏不含期号与备注',
          all(set(r) == {'bets', 'wins', 'cost', 'payout'}
              for r in AE.anonymize(records)))
    check('外部审计:报告标注外部数据源',
          '外部投注记录 vs' in AE.format_external_report(result))
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        tpl = tmpd / 'tpl.csv'
        AE.write_template(tpl)
        check('外部审计:模板可读回', len(AE.load_records(tpl)) == 2)
        badf = tmpd / 'bad.csv'
        badf.write_text('mode,bets,wins,cost,payout\nUnknown,5,0,10,0\n',
                        encoding='utf-8')
        try:
            AE.load_records(badf)
            row_rejected = False
        except ValueError:
            row_rejected = True
        check('外部审计:非法行被拒绝', row_rejected)
        dupf = tmpd / 'dup.jsonl'
        dupf.write_text('\n'.join(json.dumps(r, ensure_ascii=False)
                                  for r in records), encoding='utf-8')
        check('外部审计:JSONL 与 CSV 同结构',
              len(AE.load_records(dupf)) == 3)

    # ---- 自检钩子与发布资产 ----
    items = PC.run_checks()
    check('项目自检:6 项静态检查全通过',
          len(items) >= 6 and all(i['ok'] for i in items),
          str([(i['name'], i['ok']) for i in items if not i['ok']]))
    check('发布资产:英文README/变更日志/构建指南/模板齐全',
          all((BASE / f).is_file() for f in
              ('README.en.md', 'CHANGELOG.md', 'RELEASE_NOTES.md', 'BUILD.md',
               'records_template.csv', 'main_lite.py', 'build_exe.py',
               '.pre-commit-config.yaml')))
    check('构建脚本:lite 模式显式排除 TensorFlow',
          'EXCLUDE_LITE' in (BASE / 'build_exe.py').read_text(encoding='utf-8')
          and 'tensorflow' in (BASE / 'build_exe.py').read_text(encoding='utf-8'))
    prec = reproduce.Recorder()
    reproduce.check_game_profile(prec)
    check('复现:玩法规格声明逐项一致',
          prec.passed and len(prec.items) >= 5,
          f'{sum(1 for i in prec.items if i["ok"])}/{len(prec.items)}')


def main():
    global QUIET
    parser = argparse.ArgumentParser(description='统计核心测试（CI 用，零 TF / 零 tkinter）')
    parser.add_argument('--json', action='store_true', help='输出 JSON 结果')
    args = parser.parse_args()
    QUIET = args.json

    for fn in (test_holm_math, test_randomness, test_randomness_generic,
               test_evaluation, test_audit, test_settlement, test_data_source,
               test_aggregate_and_reproduce, test_new_features,
               test_dependency_boundary):
        try:
            fn()
        except Exception as e:
            check(f'{fn.__name__} 未抛异常', False, f'{type(e).__name__}: {e}')

    n_ok = sum(1 for _, ok in RESULTS if ok)
    passed = n_ok == len(RESULTS)
    if args.json:
        print(json.dumps({'passed': passed, 'total': len(RESULTS), 'passed_count': n_ok,
                          'items': [{'name': n, 'ok': ok} for n, ok in RESULTS]},
                         ensure_ascii=False, indent=2))
        return 0 if passed else 1

    print('-' * 70)
    print(f'统计核心测试：共 {len(RESULTS)} 项，通过 {n_ok} 项，'
          f'失败 {len(RESULTS) - n_ok} 项')
    if not passed:
        print('失败项：' + '、'.join(n for n, ok in RESULTS if not ok))
    print('说明：全量 GUI / 模型测试由本地 test_pipeline.py 负责。')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
