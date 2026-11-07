"""开奖数据随机性检验（玩法无关）

给模型一个前置判断：这批数据里到底有没有可被学习的结构。
如果连「独立均匀随机」都不能拒绝，那么任何预测模型的理论命中率上限
就是 k/B（k 为投注注数，B 为全部等概率结果数），再复杂的网络也只是在拟合噪声。

检验覆盖三个层面：
- 边际分布：各位数字是否均匀
- 横截面依赖：各位之间是否独立
- 时间依赖：期与期之间是否独立、重复号形态与和值是否符合理论分布

参数化：`analyze(data, digits, base)` 支持任意「digits 位、每位取值 [0, base)」的
等概率数字型玩法；默认 digits=3 / base=10 即排列3，输出与旧版逐位一致。
例如 `(digits=2, base=6)` 对应 36 种结果，直接：

    analyze(data, digits=2, base=6)
    analyze(data, digits=2, base=6)['n_outcomes']   # 36

检验项数随配置变化：digits 项边际均匀性 + 1 项联合独立性 +
digits 项期际独立性 + 1 项重复形态 + 1 项和值分布 = 2*digits+3。
digits=1 时联合独立性与重复形态的自由度退化为 0，自动跳过（共 3 项）。
理论分布由整数计数精确推出：重复形态用第二类 Stirling 数，和值用卷积，
不做浮点累加，也不剔除小期望格子（那会让统计量与自由度同时失真）。

多个检验构成同一个检验族，逐项以 α=0.05 裸判会产生累积误判：
默认九项时，即使数据完全随机，至少一项 p<0.05 的概率也有 1-0.95^9 ≈ 37%。
因此整体结论一律以 Holm-Bonferroni 校正后的 p 值为准（控制族错误率 FWER）。
单项原始 p 值仍保留，供诊断使用，但不作为「数据有结构」的证据。

独立运行（默认读 config 指定的排列3 数据）：

    python randomness.py
    python randomness.py --profile d2-b6 --data 其他数据.csv
    python randomness.py --digits 2 --base 6 --json > result.json
    python randomness.py --list-profiles

自定义数据支持 .xlsx / .csv（Numbers 列为空格分隔的数字串）与 .npy
（形状 (期数, digits) 的整数数组）。命令行参数只改变「数据规格」，
不改变统计量定义；非法规格由 analyze() 直接拒绝。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ALPHA = 0.05
DEFAULT_DIGITS = 3
DEFAULT_BASE = 10
# 联合列联表需要 base**digits 个格子；超过这个规模应改用专门的稀疏检验
MAX_OUTCOMES = 1_000_000

# 玩法模板：只有 arrange3 对应仓库自带数据，其余是「等概率数字型玩法」的
# 通用规格模板，用来演示参数化能力，不声称对应任何具体彩票品种。
PROFILES = {
    'arrange3': {'digits': 3, 'base': 10,
                 'label': '3 位 / 每位 0-9（本项目排列3 数据）'},
    'd1-b10': {'digits': 1, 'base': 10, 'label': '1 位 / 0-9，最小退化用例'},
    'd2-b6': {'digits': 2, 'base': 6, 'label': '2 位 / 每位 0-5，36 种结果'},
    'd2-b10': {'digits': 2, 'base': 10, 'label': '2 位 / 每位 0-9，100 种结果'},
    'd4-b10': {'digits': 4, 'base': 10, 'label': '4 位 / 每位 0-9，10000 种结果'},
}


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


def _stirling2_row(n):
    """第二类 Stirling 数 S(n, k)，返回 k=0..n 的精确整数列表。"""
    cur = [1] + [0] * n          # S(0, 0) = 1
    for m in range(1, n + 1):
        nxt = [0] * (n + 1)
        for k in range(1, m + 1):
            nxt[k] = cur[k - 1] + k * cur[k]
        cur = nxt
    return cur


def outcome_profile(digits, base):
    """按「不同取值个数 r」分组的组合计数（r 降序），返回 ([(r, count)], B)。

    count(r) = C(base, r) × S(digits, r) × r!：先选 r 个取值，再把 digits 个
    位置划成 r 个非空组（每组取同一个值），最后把这 r 个值分配到组上。
    默认 (3, 10) 得到 720 / 270 / 10，即组6 / 组3 / 豹子。
    """
    from math import comb, factorial
    stirling = _stirling2_row(digits)
    total = base ** digits
    profile = [(r, comb(base, r) * stirling[r] * factorial(r))
               for r in range(min(digits, base), 0, -1)]
    if sum(c for _, c in profile) != total:
        raise RuntimeError('重复号形态的理论计数与结果空间不一致')
    return profile, total


def sum_distribution(digits, base):
    """digits 位 [0, base) 之和的概率分布，长度 digits*(base-1)+1。

    用整数卷积精确计数后一次性归一化，避免浮点累加带来的微小漂移。
    """
    poly = [1]
    for _ in range(digits):
        nxt = [0] * (len(poly) + base - 1)
        for i, c in enumerate(poly):
            for v in range(base):
                nxt[i + v] += c
        poly = nxt
    return np.asarray(poly, dtype=float) / (base ** digits)


def analyze(data, digits=DEFAULT_DIGITS, base=DEFAULT_BASE):
    """data: (N, digits) 整数数组，每位 [0, base)，时间升序（旧 -> 新）。

    返回结构化检验结果；默认 (digits=3, base=10) 与排列3 旧版逐项一致。
    """
    if not isinstance(digits, (int, np.integer)) or not isinstance(base, (int, np.integer)):
        raise ValueError('digits / base 必须是整数')
    digits, base = int(digits), int(base)
    if digits < 1 or base < 2:
        raise ValueError('digits 必须 >= 1，base 必须 >= 2')
    outcomes = base ** digits
    if outcomes > MAX_OUTCOMES:
        raise ValueError(f'结果空间 {outcomes} 超过上限 {MAX_OUTCOMES}，'
                         '请减小 digits / base 或改用稀疏检验')

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != digits:
        raise ValueError(f'data 必须是 (N, {digits}) 的开奖号码矩阵')
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError('data 必须是整数矩阵')
    data = data.astype(np.int64, copy=False)
    n = len(data)
    if n < 100:
        raise ValueError(f'样本量过小（{n} 期），统计检验无意义')
    if data.min() < 0 or data.max() >= base:
        raise ValueError(f'开奖数字必须落在 [0, {base})')

    tests = []

    def add(name, stat, dof, p, note=''):
        tests.append({'name': name, 'stat': float(stat), 'dof': int(dof),
                      'p': float(p), 'note': note,
                      'pass': bool(p > ALPHA)})

    # ---- 1~digits. 各位数字均匀性 ----
    for pos in range(digits):
        counts = np.bincount(data[:, pos], minlength=base).astype(float)
        expected = np.full(base, n / base)
        stat = float(np.sum((counts - expected) ** 2 / expected))
        add(f'第{pos + 1}位数字均匀性', stat, base - 1, _chi2_sf(stat, base - 1),
            '最大偏差 %d 次' % int(np.abs(counts - expected).max()))

    # ---- 各位相互独立性（多维列联表完全独立检验）----
    # dof = IJK... - 1 - digits*(base-1)；默认 1000 - 1 - 27 = 972。
    # 注意不能剔除小期望格子——那样统计量与自由度会同时失真，
    # 自由度下降更快，p 值会被系统性压低到误判。
    if digits >= 2:
        grid = np.zeros((base,) * digits, dtype=float)
        np.add.at(grid, tuple(data.T), 1.0)
        indep = np.bincount(data[:, 0], minlength=base).astype(float) / n
        for i in range(1, digits):
            indep = np.multiply.outer(
                indep, np.bincount(data[:, i], minlength=base).astype(float) / n)
        obs = grid.reshape(-1)
        expect = (indep * n).reshape(-1)
        keep = expect > 0
        stat = float(np.sum((obs[keep] - expect[keep]) ** 2 / expect[keep]))
        dof = outcomes - 1 - digits * (base - 1)
        name = ('三位数字相互独立性' if digits == 3
                else f'{digits}位数字相互独立性')
        add(name, stat, dof, _chi2_sf(stat, dof),
            f'{outcomes} 种组合已出现 %d 种' % int((obs > 0).sum()))

    # ---- 期际独立性（上期 -> 本期转移矩阵）----
    for pos in range(digits):
        trans = np.zeros((base, base))
        np.add.at(trans, (data[:-1, pos], data[1:, pos]), 1.0)
        row_sum = trans.sum(axis=1, keepdims=True)
        col_sum = trans.sum(axis=0, keepdims=True)
        expect = row_sum * col_sum / trans.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = float(np.nansum((trans - expect) ** 2 / expect))
        dof = (base - 1) ** 2
        add(f'第{pos + 1}位期际独立性', stat, dof, _chi2_sf(stat, dof))

    # ---- 重复号形态：按「不同取值个数」分组 ----
    if digits >= 2:
        profile, _ = outcome_profile(digits, base)
        sorted_rows = np.sort(data, axis=1)
        r_obs = 1 + np.count_nonzero(
            sorted_rows[:, 1:] != sorted_rows[:, :-1], axis=1)
        obs3 = np.array([(r_obs == r).sum() for r, _ in profile], dtype=float)
        theo3 = np.array([c / outcomes for _, c in profile]) * n
        stat = float(np.sum((obs3 - theo3) ** 2 / theo3))
        name = ('重复号形态(组6/组3/豹子)' if (digits, base) == (3, 10)
                else '重复号形态(按不同取值数)')
        add(name, stat, len(profile) - 1, _chi2_sf(stat, len(profile) - 1),
            '观测 %s' % (obs3 / n).round(4).tolist())

    # ---- 和值分布 ----
    theo = sum_distribution(digits, base)
    sums = data.sum(axis=1)
    obs_sum = np.bincount(sums, minlength=len(theo)).astype(float)
    expect_sum = theo * n
    stat = float(np.sum((obs_sum - expect_sum) ** 2 / np.maximum(expect_sum, 1e-9)))
    dof = digits * (base - 1)
    add('和值分布', stat, dof, _chi2_sf(stat, dof),
        '均值 %.3f（理论 %.1f）' % (sums.mean(), digits * (base - 1) / 2))

    # ---- 多重比较校正：同族检验的整体结论必须控制 FWER ----
    p_holm = holm_adjust([t['p'] for t in tests])
    for t, adj in zip(tests, p_holm):
        t['p_holm'] = float(adj)
        t['pass_holm'] = bool(adj > ALPHA)

    m = len(tests)
    n_nominal = sum(1 for t in tests if not t['pass'])
    n_holm = sum(1 for t in tests if not t['pass_holm'])
    return {
        'n': n, 'tests': tests, 'alpha': ALPHA, 'n_tests': m,
        'digits': digits, 'base': base, 'n_outcomes': outcomes,
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
    outcomes = result.get('n_outcomes', DEFAULT_BASE ** DEFAULT_DIGITS)
    spec = (f"{result.get('digits', DEFAULT_DIGITS)} 位 × "
            f"{result.get('base', DEFAULT_BASE)} 进制")
    lines = ['=' * 74,
             f"开奖数据随机性检验（{result['n']} 期，{spec}，{n_tests} 项，Holm 校正）",
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
        # 显著分支同样给出结果空间，但用条件句：此时不能再断言「概率恒为 k/B」
        lines.append(f'结果空间 B={outcomes}；若随机假设成立，k 注互不重复的单期'
                     f'中奖概率上限为 k/{outcomes}。')
    elif n_nominal:
        lines.append(f"裸判有 {n_nominal} 项 p<0.05，但 Holm 校正后均不显著——"
                     '属于多重比较下的偶然波动，不构成「数据有结构」的证据。')
        lines.append('整体结论：数据与「独立均匀随机」假设相容。')
        lines.append('含义：不存在可被模型利用的结构。任何选号方式的单期中奖概率')
        lines.append(f'都是 k/{outcomes}（k 为互不重复的注数），与选号策略无关。')
    else:
        lines.append(f"全部 {n_tests} 项检验原始 p 与 Holm 校正后 p 均 > "
                     f"{result['alpha']}，数据与「独立均匀随机」假设完全相容。")
        lines.append('含义：不存在可被模型利用的结构。任何选号方式的单期中奖概率')
        lines.append(f'都是 k/{outcomes}（k 为互不重复的注数），与选号策略无关。')
    lines.append('=' * 74)
    return '\n'.join(lines)


def _parse_rows(rows, digits, base):
    """把「空格分隔的数字串」序列转成 (n, digits) 整数矩阵并校验取值。"""
    out = []
    for i, raw in enumerate(rows):
        tokens = str(raw).replace(',', ' ').split()
        if len(tokens) != digits:
            raise ValueError(f'第 {i + 1} 行应有 {digits} 个数字，实际 {len(tokens)} 个')
        try:
            vals = [int(t) for t in tokens]
        except ValueError:
            raise ValueError(f'第 {i + 1} 行含非整数：{raw!r}')
        if any(v < 0 or v >= base for v in vals):
            raise ValueError(f'第 {i + 1} 行数字越界，允许 [0, {base})：{vals}')
        out.append(vals)
    return np.asarray(out, dtype=int).reshape(-1, digits)


def load_draws(path=None, digits=DEFAULT_DIGITS, base=DEFAULT_BASE,
               numbers_col='Numbers'):
    """读取开奖数据 → (期数, digits) 整数矩阵。

    - None / .xlsx：读 config 指定的工作簿，按 Period 升序排列
    - .csv：读指定列（默认 Numbers），每行是空格分隔的数字串
    - .npy：直接加载形状为 (期数, digits) 的整数数组
    """
    if path is None:
        from paths import resolve
        path = resolve('历史数据.xlsx')
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.npy':
        data = np.asarray(np.load(path))
    elif suffix == '.xlsx':
        import pandas as pd
        df = pd.read_excel(path, engine='openpyxl')
        if numbers_col not in df.columns:
            raise ValueError(f'工作簿缺少列 {numbers_col}，现有列：{list(df.columns)}')
        if 'Period' in df.columns:
            df = df.sort_values('Period')
        data = _parse_rows(df[numbers_col].tolist(), digits, base)
    elif suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or numbers_col not in reader.fieldnames:
                raise ValueError(f'CSV 缺少列 {numbers_col}，现有列：{reader.fieldnames}')
            data = _parse_rows([r[numbers_col] for r in reader], digits, base)
    else:
        raise ValueError(f'不支持的数据格式：{suffix}（支持 .xlsx / .csv / .npy）')

    if data.ndim != 2 or data.shape[1] != digits:
        raise ValueError(f'数据形状应为 (期数, {digits})，实际 {data.shape}')
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError('数据必须是整数')
    if data.size and (data.min() < 0 or data.max() >= base):
        raise ValueError(f'数字越界，允许 [0, {base})')
    return data


def _json_default(obj):
    """让 numpy 标量/数组能直接进 json.dumps。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f'无法序列化：{type(obj).__name__}')


def resolve_spec(args):
    """命令行参数 → (digits, base)；显式参数优先于玩法模板。"""
    digits, base = DEFAULT_DIGITS, DEFAULT_BASE
    if args.profile:
        if args.profile not in PROFILES:
            raise ValueError(f'未知玩法模板 {args.profile!r}，可用：'
                             f'{", ".join(sorted(PROFILES))}')
        digits = PROFILES[args.profile]['digits']
        base = PROFILES[args.profile]['base']
    if args.digits is not None:
        digits = args.digits
    if args.base is not None:
        base = args.base
    return digits, base


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='开奖数据随机性检验（玩法无关）')
    parser.add_argument('--data', default=None,
                        help='数据文件：.xlsx / .csv / .npy；默认读仓库自带工作簿')
    parser.add_argument('--numbers-col', default='Numbers', help='开奖号所在列名')
    parser.add_argument('--profile', default=None, help='玩法模板名')
    parser.add_argument('--digits', type=int, default=None, help='位数（覆盖模板）')
    parser.add_argument('--base', type=int, default=None, help='每位取值个数（覆盖模板）')
    parser.add_argument('--list-profiles', action='store_true', help='列出玩法模板并退出')
    parser.add_argument('--json', action='store_true', help='输出结构化 JSON 而非文本报告')
    args = parser.parse_args()

    if args.list_profiles:
        for name, spec in PROFILES.items():
            print(f"{name:<10} digits={spec['digits']} base={spec['base']}  "
                  f"结果空间={spec['base'] ** spec['digits']:<8} {spec['label']}")
        return 0

    try:
        digits, base = resolve_spec(args)
        data = load_draws(args.data, digits, base, args.numbers_col)
        result = analyze(data, digits=digits, base=base)
    except (ValueError, OSError) as e:
        print(f'参数或数据错误：{e}', file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(format_report(result))
    return 0 if result['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
