"""严格时序回测与随机基线对照。

核心原则：预测第 t 期时只允许看到 data[:t]，绝不接触未来数据。
所有"优化手段"都在这里和随机选号放在同一把尺子下比较：
逐位命中率、直选命中、组选命中，配 Wilson 置信区间与二项检验。

独立运行：
    python evaluation.py                 # 全量回测（轻量策略）
    python evaluation.py --with-ml       # 追加逻辑回归 / MLP「调模型」对照
    python evaluation.py --n 3000        # 只用最近 3000 期
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

BASE_DIR = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# 理论基线（均匀独立、每位 0-9）
P_DIGIT = 0.1                 # 单个位置猜中的概率
P_EXACT = 0.001               # 直选 3 位全中
P_SET = 0.00514               # 组选（忽略顺序、保留重复）猜中的概率

# 体彩排列3 现行规则（单注 2 元，实际以销售当期官方公告为准）：
# 直选 1040 元 / 组选3 346 元 / 组选6 173 元。
# hit_per_bet 为单注中奖概率：直选 1/1000；组选3 押一对重复数字 3/1000；
# 组选6 押三个不同数字 6/1000。三种玩法返奖率同为约 52%——
# 玩法只改变「命中频率」，不改变「长期是亏的」这个事实。
PLAY_RULES = {
    '直选': {'payout': 1040.0, 'hit_per_bet': 1 / 1000, 'max_bets': 1000},
    '组选3': {'payout': 346.0, 'hit_per_bet': 3 / 1000, 'max_bets': 90},
    '组选6': {'payout': 173.0, 'hit_per_bet': 6 / 1000, 'max_bets': 120},
}


def canonical(nums):
    """把 '9 4 4' / '944' / [9, 4, 4] 统一为排序后的 3 字符数字串 '449'。"""
    if isinstance(nums, str):
        parts = nums.split() if ' ' in nums else list(nums)
    else:
        parts = list(nums)
    if len(parts) != 3:
        raise ValueError(f'需要 3 个数字，得到 {nums!r}')
    try:
        digits = [int(p) for p in parts]
    except (TypeError, ValueError):
        raise ValueError(f'无法解析为数字：{nums!r}')
    if any(not 0 <= d <= 9 for d in digits):
        raise ValueError(f'数字须在 0~9：{nums!r}')
    return ''.join(str(d) for d in sorted(digits))


def bet_hits(bet, actual, mode):
    """单注 bet 是否命中开奖 actual（均为 3 位数字，actual 可乱序或含空格）。

    直选：顺序敏感相等；组选6：数字相同（不限顺序）且三位互不相同；
    组选3：数字相同且恰有一对相同。mode 非法或格式非法抛 ValueError。
    """
    if mode not in PLAY_RULES:
        raise ValueError(f'未知玩法：{mode}')
    bet_c = canonical(bet)                # 一次调用同时完成合法性校验
    actual_c = canonical(actual)
    if mode == '直选':
        bet_str = bet if isinstance(bet, str) else ''.join(map(str, bet))
        actual_str = actual if isinstance(actual, str) else ''.join(map(str, actual))
        return bet_str.replace(' ', '') == actual_str.replace(' ', '')
    distinct = len(set(bet_c))
    if mode == '组选6':
        return bet_c == actual_c and distinct == 3
    return bet_c == actual_c and distinct == 2


def play_odds(mode, bets, stake_per_bet=2.0):
    """bets 注互不重复组合在指定玩法下的理论账目。

    覆盖率与选号方式无关：只要 bets 注互不重复，
    单期中奖概率 = bets × 单注中奖概率。
    """
    rule = PLAY_RULES[mode]
    bets = int(bets)
    if not 1 <= bets <= rule['max_bets']:
        raise ValueError(f'{mode} 注数须在 1~{rule["max_bets"]} 之间')
    coverage = bets * rule['hit_per_bet']
    cost = bets * stake_per_bet
    expected = coverage * rule['payout']
    return {
        'mode': mode,
        'bets': bets,
        'payout': rule['payout'],
        'coverage': coverage,
        'coverage_pct': coverage * 100,
        'one_in': 1 / coverage if coverage else float('inf'),
        'cost': cost,
        'expected_return': expected,
        'expected_profit': expected - cost,
        'return_rate': expected / cost,
    }


def budget_plan(budget, stake_per_bet=2.0):
    """给定预算，输出三种玩法的命中概率与期望对照。"""
    bets = int(budget // stake_per_bet)
    if bets < 1:
        raise ValueError('预算不足一注')
    return {'bets': bets,
            'modes': {m: play_odds(m, bets, stake_per_bet) for m in PLAY_RULES}}


def multi_bet_odds(k, stake_per_bet=2.0, payout_per_hit=None):
    """k 注互不重复直选的理论账目（play_odds 的直选快捷方式）。

    payout_per_hit 默认 1040 元：体彩排列3 现行规则的直选单注固定奖金
    （当期奖金计提 52%、调节基金 1%）。仅用于量级估算，
    实际以销售当期官方公告为准。
    """
    odds = play_odds('直选', k, stake_per_bet)
    if payout_per_hit is not None and payout_per_hit != odds['payout']:
        coverage = k * P_EXACT
        cost = k * stake_per_bet
        expected = coverage * payout_per_hit
        odds.update(expected_return=expected,
                    expected_profit=expected - cost,
                    return_rate=expected / cost)
    return odds


def wilson_interval(k, n, z=1.96):
    """二项比例的 Wilson 95% 置信区间，小样本也比正态近似可靠。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def longest_false_streak(flags):
    """最长连续失败期数。全 False 返回 len(flags)，全 True 返回 0。

    回测、模拟与对账共用的单一实现（此前 simulation/settlement 各有一份）。
    """
    streak = best = 0
    for f in flags:
        streak = 0 if f else streak + 1
        best = max(best, streak)
    return best


class Strategy:
    name = 'base'

    def predict(self, history):
        """history: (t, 3) 时间升序，截止当前期之前。返回 3 个 0-9 的数字。"""
        raise NotImplementedError

    def observe(self, actual):
        """回测框架在揭晓当期后调用，可用于增量更新（仍严格只用已发生数据）。"""


class RandomStrategy(Strategy):
    name = '随机选号(基线)'

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, history):
        return self.rng.integers(0, 10, 3)


class FixedStrategy(Strategy):
    name = '固定号码 000'

    def predict(self, history):
        return np.array([0, 0, 0])


class _CountingStrategy(Strategy):
    """按位维护 0-9 计数的热/冷号基类。

    计数在首次 predict 时用当时可见的历史一次性建立（O(N)），之后由
    observe 增量维护——run_backtest 保证每期揭晓后调用 observe，因此
    总复杂度 O(N)，等价于旧的逐期全量 bincount（O(N²)）。
    """

    pick_best = True   # True=热号(argmax)，False=冷号(argmin)

    def __init__(self):
        self.counts = None

    def predict(self, history):
        if self.counts is None:
            self.counts = np.stack([np.bincount(history[:, i], minlength=10)
                                    for i in range(3)]).astype(np.int64)
        chooser = np.argmax if self.pick_best else np.argmin
        return np.array([int(chooser(self.counts[i])) for i in range(3)])

    def observe(self, actual):
        if self.counts is None:
            return
        actual = np.asarray(actual)
        for i in range(3):
            self.counts[i, int(actual[i])] += 1


class HotStrategy(_CountingStrategy):
    name = '热号(每位历史最高频)'
    pick_best = True


class ColdStrategy(_CountingStrategy):
    name = '冷号(每位历史最低频)'
    pick_best = False


class LastDrawStrategy(Strategy):
    name = '照抄上一期'

    def predict(self, history):
        return history[-1].copy()


class MarkovStrategy(Strategy):
    name = '马尔可夫转移(上期→下期)'

    def __init__(self):
        self.counts = np.ones((3, 10, 10))
        self.prev = None

    def predict(self, history):
        if self.prev is None:
            return history[-1].copy()
        return np.array([self.counts[i, self.prev[i]].argmax() for i in range(3)])

    def observe(self, actual):
        actual = np.asarray(actual)
        if self.prev is not None:
            for i in range(3):
                self.counts[i, self.prev[i], actual[i]] += 1
        self.prev = actual.copy()


class EnsembleStrategy(Strategy):
    name = '集成投票(热号+马尔可夫+上期)'

    def __init__(self):
        self.members = [HotStrategy(), MarkovStrategy(), LastDrawStrategy()]

    def predict(self, history):
        votes = np.stack([np.asarray(m.predict(history)) for m in self.members])
        return np.array([np.bincount(votes[:, i], minlength=10).argmax()
                         for i in range(3)])

    def observe(self, actual):
        for m in self.members:
            m.observe(actual)


def _window_features(history, window=60):
    """从最近 window 期提取 39 维特征（频率、lag 匹配、上期形态、上期号码）。"""
    w = history[-window:]
    feats = []
    for i in range(3):
        feats.extend(np.bincount(w[:, i], minlength=10) / len(w))
    feats.extend((w[-1] == w[-2]).astype(float))
    feats.extend([w[-1].mean() / 9.0, (w[-1] % 2).mean(), (w[-1] >= 5).mean()])
    feats.extend(w[-1] / 9.0)
    return np.array(feats, dtype=np.float32)


class SklearnStrategy(Strategy):
    """「加特征 + 换模型」的代表：只在回测起点之前的数据上训练一次，严格 holdout。"""

    def __init__(self, kind='logistic', train_size=3000, window=60, random_state=0):
        self.name = ('逻辑回归(39维特征)' if kind == 'logistic'
                     else 'MLP神经网络(39维特征)')
        self.kind = kind
        self.train_size = train_size
        self.window = window
        self.random_state = random_state
        self.models = None

    def _fit(self, history):
        import warnings
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        train = history[:self.train_size]
        X = np.array([_window_features(train[:t], self.window)
                      for t in range(self.window + 1, len(train))],
                     dtype=np.float32)
        y = train[self.window + 1:]
        is_mlp = self.kind == 'mlp'
        self.models = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(3):
                model = (MLPClassifier(hidden_layer_sizes=(64,), max_iter=40,
                                       random_state=self.random_state)
                         if is_mlp else LogisticRegression(max_iter=120, C=0.1))
                model.fit(X, y[:, i])
                self.models.append(model)

    def predict(self, history):
        if self.models is None:
            self._fit(history)
        x = _window_features(history, self.window).reshape(1, -1)
        return np.array([int(m.predict(x)[0]) for m in self.models])


def default_strategies(with_ml=False, train_size=3000):
    builders = [lambda: RandomStrategy(0), FixedStrategy, HotStrategy,
                ColdStrategy, LastDrawStrategy, MarkovStrategy,
                EnsembleStrategy]
    if with_ml:
        builders += [lambda: SklearnStrategy('logistic', train_size),
                     lambda: SklearnStrategy('mlp', train_size)]
    return [b() for b in builders]


def assess(digit_hits, exact_hits, set_hits, n):
    """把命中数换算成率、置信区间与显著性。"""
    d_lo, d_hi = wilson_interval(digit_hits, 3 * n)
    e_lo, e_hi = wilson_interval(exact_hits, n)
    s_lo, s_hi = wilson_interval(set_hits, n)
    return {
        'n': n,
        'digit_hits': digit_hits,
        'digit_rate': digit_hits / (3 * n) if n else 0.0,
        'digit_ci': (d_lo, d_hi),
        'digit_p': float(binomtest(digit_hits, 3 * n, P_DIGIT).pvalue) if n else 1.0,
        'exact_hits': exact_hits,
        'exact_rate': exact_hits / n if n else 0.0,
        'exact_ci': (e_lo, e_hi),
        'exact_p': float(binomtest(exact_hits, n, P_EXACT).pvalue) if n else 1.0,
        'set_hits': set_hits,
        'set_rate': set_hits / n if n else 0.0,
        'set_ci': (s_lo, s_hi),
    }


def run_backtest(data, strategies=None, start=3000, with_ml=False):
    """逐期滚动回测。返回 {策略名: 评估指标}。

    start 之前的数据只作为历史/训练素材；评估区间为 [start, len(data))。
    """
    data = np.asarray(data, dtype=np.int32)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError('data 必须是 (N, 3)')
    if start < 60:
        raise ValueError('start 至少 60，保证有足够历史')
    if start >= len(data):
        raise ValueError('start 超出数据长度')
    if strategies is None:
        strategies = default_strategies(with_ml=with_ml, train_size=start)

    n = len(data) - start
    out = {}
    for s in strategies:
        digit_hits = exact_hits = set_hits = 0
        for t in range(start, len(data)):
            pred = np.asarray(s.predict(data[:t]), dtype=np.int32)
            actual = data[t]
            if pred.shape != (3,) or not np.all((pred >= 0) & (pred <= 9)):
                raise ValueError(f'{s.name} 返回非法预测：{pred}')
            digit_hits += int((pred == actual).sum())
            exact_hits += int(np.array_equal(pred, actual))
            set_hits += int(np.array_equal(np.sort(pred), np.sort(actual)))
            s.observe(actual)
        out[s.name] = assess(digit_hits, exact_hits, set_hits, n)
    return out


def bonferroni_threshold(results):
    """同时检验多个策略时，用 Bonferroni 校正避免「总有一个显著」的假象。"""
    return 0.05 / max(1, len(results))


def _pad(text, width, align='left'):
    """按显示宽度对齐（中文算 2 列），避免控制台表格错位。"""
    w = sum(2 if ord(c) > 127 else 1 for c in str(text))
    gap = ' ' * max(1, width - w)
    return f'{text}{gap}' if align == 'left' else f'{gap}{text}'


def format_report(results, title='时序回测报告'):
    n = next(iter(results.values()))['n']
    alpha = bonferroni_threshold(results)
    expected_exact = n * P_EXACT
    head = (_pad('策略', 30) + _pad('逐位命中率', 12, 'right')
            + _pad('95% CI', 20, 'right') + _pad('p(逐位)', 10, 'right')
            + _pad('直选/期望', 14, 'right') + _pad('p(直选)', 10, 'right')
            + _pad('组选', 8, 'right'))
    lines = [title,
             f"评估 {n} 期 | 逐位基线 {P_DIGIT:.4f} | 直选基线 {P_EXACT:.4f} "
             f"（期望 {expected_exact:.1f} 次） | 组选基线 {P_SET:.5f}",
             f"同时检验 {len(results)} 个策略，Bonferroni 校正后显著性阈值 p < {alpha:.5f}",
             '', head, '-' * 104]
    for name, r in results.items():
        ci = f"[{r['digit_ci'][0]:.4f}, {r['digit_ci'][1]:.4f}]"
        lines.append(
            _pad(name, 30)
            + _pad(f"{r['digit_rate']:.4f}", 12, 'right')
            + _pad(ci, 20, 'right')
            + _pad(f"{r['digit_p']:.3f}", 10, 'right')
            + _pad(f"{r['exact_hits']} / {expected_exact:.1f}", 14, 'right')
            + _pad(f"{r['exact_p']:.3f}", 10, 'right')
            + _pad(str(r['set_hits']), 8, 'right'))

    best_digit = min(results.items(), key=lambda kv: kv[1]['digit_p'])
    best_exact = max(results.items(), key=lambda kv: kv[1]['exact_hits'])
    sig = [name for name, r in results.items() if r['digit_p'] < alpha]
    lines += ['', '自动判读：',
              f"· 逐位命中率最接近「显著」的是 {best_digit[0]}（p={best_digit[1]['digit_p']:.3f}），"
              f"校正阈值 {alpha:.5f}，{'通过' if sig else '未通过'}。",
              f"· 直选命中最多的是 {best_exact[0]}：{best_exact[1]['exact_hits']} 次 "
              f"vs 随机期望 {expected_exact:.1f} 次（p={best_exact[1]['exact_p']:.3f}）；"
              f"{len(results)} 个策略里挑最好的，出现这种波动很正常。",
              f"· 校正后显著优于随机的策略：{('、'.join(sig)) if sig else '无'}。",
              '· 直选命中在数千期样本下期望仅个位数，任何「命中翻倍」都可能是噪声，'
              '请以逐位命中率的置信区间为准。']
    return '\n'.join(lines)


def load_excel_data(path=None, num_rows=0):
    """独立于 TensorFlow 读取开奖数据，返回时间升序 (N,3)。"""
    import pandas as pd

    if path is None:
        cfg = json.loads((BASE_DIR / 'config.json').read_text(encoding='utf-8'))
        path = BASE_DIR / cfg['default_file_path']
    df = pd.read_excel(path, engine='openpyxl')
    df = df.sort_values('Period', ascending=True)
    if num_rows:
        df = df.tail(num_rows)
    numbers = [str(x).split() for x in df['Numbers']]
    if any(len(r) != 3 or any(len(t) != 1 or not t.isdigit() for t in r)
           for r in numbers):
        raise ValueError('Numbers 列格式异常')
    return np.array([[int(t) for t in r] for r in numbers], dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description='严格时序回测与随机基线对照')
    ap.add_argument('--n', type=int, default=0, help='只用最近 N 期（0=全部）')
    ap.add_argument('--start', type=int, default=3000, help='回测起点（之前的数据仅作历史）')
    ap.add_argument('--with-ml', action='store_true', help='追加逻辑回归/MLP 对照')
    ap.add_argument('--json', action='store_true', help='输出 JSON 而非表格')
    args = ap.parse_args()

    data = load_excel_data(num_rows=args.n)
    start = max(60, min(args.start, len(data) // 2))
    results = run_backtest(data, start=start, with_ml=args.with_ml)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(format_report(results, title=f'时序回测报告（共 {len(data)} 期，起点 {start}）'))


if __name__ == '__main__':
    main()
