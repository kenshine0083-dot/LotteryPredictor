"""命中模拟器与模型变体对照。

回答一个问题：「多注覆盖 + 组选6」这套提高命中的方案，在真实历史
数据上表现如何？逐期回放：每期独立选注、与实际开奖判定、按玩法记账。

原则：
- 选号与开奖独立，每期成本 = 注数 × 2 元（排列3 无跨期持仓）；
- 任何选号策略已实测与随机等价（见 evaluation.run_backtest），
  因此默认用「随机不重复选号」作代表，并可选模型选号对照；
- 所有输出标注理论命中率与约 52% 的返奖率——命中更频繁，
  长期期望仍为负。

独立运行：
    python simulation.py --mode 组选6 --bets 30          # 随机选号模拟
    python simulation.py --bets 120 --mode 组选6         # 自检锚点≈0.72
    python simulation.py --selection model --bets 30     # 模型选号对照
    python simulation.py --variants                      # 模型变体回测
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import evaluation as EV
# combos 是纯计算层（无 TF 依赖）：模拟器不再从 lottery_model 顶层导入，
# 因此轻量版 GUI / 命令行回放无需加载 TensorFlow。
from combos import (COMBO_MAX_BETS, SEQUENCE_LENGTH_DEFAULT, iter_combos,
                    rank_combos, sliding_windows)

# 最长连败的单一实现在 evaluation（settlement 也复用）；保留同名导出兼容旧调用。
longest_false_streak = EV.longest_false_streak

# 模型选号模拟的数据量 = eval_len + 该预留量（保证训练集有足够历史）。
MODEL_SIM_HEADROOM = 2200

BASE_DIR = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def combo_space_size(mode):
    return COMBO_MAX_BETS[mode]


def sample_bets(rng, mode, bets):
    """在对应组合空间内均匀抽 bets 注互不重复号码（返回字符串列表）。

    直选空间为 1000 个有序组合；组选6/组选3 为无序数字组
    （120/90 个），与 predict_multiple 的口径一致。
    注数等于空间上限时直接全收。
    """
    bets = int(bets)
    size = combo_space_size(mode)
    if not 1 <= bets <= size:
        raise ValueError(f'{mode} 注数须在 1~{size} 之间')
    idx = np.arange(size)
    if bets == size:
        chosen = idx
    else:
        chosen = rng.choice(size, size=bets, replace=False)
    # 组合枚举与 predict_multiple 共用 iter_combos，保证三处口径一致。
    combos = _all_bets(mode)
    return [combos[i] for i in chosen]


def simulate_random(data, mode, bets, start=None, seed=0,
                    stake_per_bet=2.0, interrupt=None):
    """逐期回放 data[start:]，随机不重复选号 + 实际开奖判定。

    data: (N, 3) int，时间升序。返回统计 dict（见 format_simulation_report）。
    """
    rule = EV.PLAY_RULES[mode]
    bets = int(bets)
    size = combo_space_size(mode)
    if not 1 <= bets <= size:
        raise ValueError(f'{mode} 注数须在 1~{size} 之间')
    n = len(data) if start is None else min(len(data) - int(start), len(data))
    if n < 1:
        raise ValueError('评估区间为空')
    rng = np.random.default_rng(seed)
    theoretical = bets * rule['hit_per_bet']
    win_flags = []
    cum_pnl = []
    pnl = 0.0
    cost_total = 0.0
    return_total = 0.0
    for i in range(n):
        if interrupt is not None and interrupt():
            n = i
            break
        draw = ''.join(map(str, data[i]))
        picks = sample_bets(rng, mode, bets) if bets < size else None
        win = sum(1 for p in (picks if picks is not None
                              else _all_bets(mode)) if EV.bet_hits(p, draw, mode))
        cost_total += bets * stake_per_bet
        if win:
            return_total += win * rule['payout']
        pnl += win * rule['payout'] - bets * stake_per_bet
        cum_pnl.append(pnl)
        win_flags.append(win > 0)
    hit_draws = sum(win_flags)
    ci = EV.wilson_interval(hit_draws, n)
    p = float(binomtest(hit_draws, n, theoretical).pvalue) if n else 1.0
    return {
        'n': n, 'mode': mode, 'bets': bets, 'seed': seed,
        'hit_draws': hit_draws,
        'hit_rate': hit_draws / n if n else 0.0,
        'theoretical_rate': theoretical,
        'ci': ci, 'p': p,
        'expected_hits': n * theoretical,
        'longest_lose_streak': longest_false_streak(win_flags),
        'cost_total': cost_total,
        'return_total': return_total,
        'pnl': return_total - cost_total,
        'cum_pnl': cum_pnl,
        'win_flags': win_flags,
        'stake_per_bet': stake_per_bet,
        'payout': rule['payout'],
    }


@lru_cache(maxsize=None)
def _all_bets(mode):
    """全收/抽样共用的完整组合空间（按玩法缓存，避免逐期重建）。"""
    return list(iter_combos(mode))


def format_simulation_report(r):
    """模拟结果 → 文本报告（含理论对照与 52% 返奖率声明）。"""
    ci = f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
    cum = r['cum_pnl']
    lines = [
        f"命中模拟 | {r['mode']} × {r['bets']} 注 × {r['n']} 期（seed={r['seed']}，随机不重复选号）",
        f"实际命中 {r['hit_draws']} 期 | 命中率 {r['hit_rate']:.4f} | 95% CI {ci}",
        f"理论命中率 {r['theoretical_rate']:.4f}（{r['bets']}/1000）"
        f" | 理论期望命中 {r['expected_hits']:.1f} 期 | 二项检验 p={r['p']:.3f}",
        f"最长连败 {r['longest_lose_streak']} 期",
        f"总投入 {r['cost_total']:.0f} 元 | 总回报 {r['return_total']:.0f} 元"
        f" | 净损益 {r['pnl']:+.0f} 元",
    ]
    if cum:
        peak = max(cum)
        trough = min(cum)
        lines.append(f"累计收益曲线：期末 {cum[-1]:+.0f} 元 | 峰值 {peak:+.0f} 元 | 谷值 {trough:+.0f} 元")
    lines += [
        f"声明：单注奖金 {r['payout']:.0f} 元，返奖率约 52%，长期期望为负；"
        "模拟为逐期独立重买，成本按期累计。",
        "判读：命中率 CI 覆盖理论值 = 方案命中频率符合数学预期；"
        "「提升命中」指提升中奖频率，不改变亏损期望。",
    ]
    return '\n'.join(lines)


class ModelPicker:
    """单模型固定训练集滚动选号：在 data[:train_end] 上训练一次，
    之后逐期预测（特征只用 data[:t]，无未来泄漏）。

    特征逐行独立且 scaler 只在训练集上拟合一次，因此全量变换矩阵
    在 __init__ 缓存一次，pick 直接取前缀切片——旧的逐期重算
    （O(eval_len × N)）被消除，300 期评估提速约 300 倍。
    """

    def __init__(self, data_processor, train_end, seq_len=SEQUENCE_LENGTH_DEFAULT,
                 epochs=3, log=print):
        import tensorflow as tf
        from lottery_model import LotteryModel
        self.dp = data_processor
        self.seq_len = seq_len
        X = data_processor.build_feature_matrix(fit_scalers=True, upto=train_end)
        count = train_end - seq_len
        Xw = sliding_windows(X, seq_len)[:count]
        yw = data_processor.data[seq_len:train_end].astype(np.int32)
        self.model = LotteryModel(seq_len, None).build_model()
        self.model.fit(Xw, yw, epochs=epochs, batch_size=16, verbose=0)
        log(f"模型选号器：训练 {count} 样本（截至期 {train_end}），{epochs} epochs")
        self._tf = tf
        # scaler 已在 [:train_end] 上拟合；用它们变换全量数据并缓存。
        # matrix[:t] == build_feature_matrix(fit_scalers=False, upto=t)。
        self.matrix = data_processor.build_feature_matrix(fit_scalers=False)

    def pick(self, t, mode, k):
        """预测第 t 期：特征只用 data[:t]。返回 [{'numbers','score'}]。"""
        window = self.matrix[:t][-self.seq_len:]
        probs = self.model.predict(window.reshape(1, self.seq_len, -1), verbose=0)[0]
        return rank_combos(probs, mode, k)


def simulate_model(dp, mode, bets, eval_len=300, epochs=3,
                   seq_len=SEQUENCE_LENGTH_DEFAULT, log=print, interrupt=None):
    """模型选号版模拟：固定训练集 + 最近 eval_len 期逐期选号判定。

    模型只决定「选哪 k 注」；只要不重复，命中率与随机选号完全相同。
    """
    data = dp.data
    train_end = len(data) - eval_len
    if train_end <= seq_len + 10:
        raise ValueError('数据不足，无法留出评估区间')
    rule = EV.PLAY_RULES[mode]
    picker = ModelPicker(dp, train_end, seq_len=seq_len, epochs=epochs, log=log)
    theoretical = bets * rule['hit_per_bet']
    win_flags = []
    cost_total = 0.0
    return_total = 0.0
    for i, t in enumerate(range(train_end, len(data))):
        if interrupt is not None and interrupt():
            break
        if i % 50 == 0:
            log(f"模型选号模拟进度 {i}/{eval_len}")
        draw = ''.join(map(str, data[t]))
        picks = picker.pick(t, mode, bets)
        win = sum(1 for p in picks if EV.bet_hits(p['numbers'], draw, mode))
        cost_total += bets * 2.0
        if win:
            return_total += win * rule['payout']
        win_flags.append(win > 0)
    n = len(win_flags)
    hit_draws = sum(win_flags)
    return {
        'n': n, 'mode': mode, 'bets': bets,
        'hit_draws': hit_draws,
        'hit_rate': hit_draws / n if n else 0.0,
        'theoretical_rate': theoretical,
        'ci': EV.wilson_interval(hit_draws, n),
        'p': float(binomtest(hit_draws, n, theoretical).pvalue) if n else 1.0,
        'expected_hits': n * theoretical,
        'longest_lose_streak': longest_false_streak(win_flags),
        'cost_total': cost_total,
        'return_total': return_total,
        'pnl': return_total - cost_total,
        'payout': rule['payout'],
    }


# ---- 模型变体对照（「调模型」的如实验证）----

def _variant_feature_matrix(data):
    """(N,9)：号码/9 + lag匹配数/3 + 奇比 + 大比 + 和值/27。逐行独立 + 前一期。"""
    d = data.astype(np.float32)
    norm = d / 9.0
    mc = np.zeros_like(d)
    if len(d) > 1:
        mc[1:] = (d[1:] == d[:-1]).sum(axis=1, keepdims=True)
    mc /= 3.0
    odd = ((d % 2 == 1).sum(axis=1, keepdims=True) / 3.0)
    big = ((d >= 5).sum(axis=1, keepdims=True) / 3.0)
    s = (d.sum(axis=1, keepdims=True) / 27.0)
    return np.hstack([norm, mc, odd, big, s]).astype(np.float32)


def _variant_features(data, variant):
    base = _variant_feature_matrix(data)
    if variant == 'sumspan':
        span = (data.max(axis=1) - data.min(axis=1)).astype(np.float32) / 9.0
        return np.hstack([base, span.reshape(-1, 1)]).astype(np.float32)
    return base


class VariantStrategy:
    """自建小型模型（独立于主流程 Transformer 与 9 维约定）。

    变体：window100 = 序列长度 100；sumspan = 基础 9 维 + 归一化和值/跨度。
    """

    def __init__(self, data, variant, train_end, epochs=3, log=print):
        import tensorflow as tf
        self.variant = variant
        self.seq_len = 100 if variant == 'window100' else 50
        feats = _variant_features(data, variant)
        dim = feats.shape[1]
        count = train_end - self.seq_len
        if count < 10:
            raise ValueError('训练区间过短')
        X = np.array([feats[i:i + self.seq_len] for i in range(count)], dtype=np.float32)
        y = data[self.seq_len:train_end].astype(np.int32)
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.seq_len, dim)),
            tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(30),
            tf.keras.layers.Reshape((3, 10)),
            tf.keras.layers.Softmax(axis=-1),
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='digit_acc')])
        model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
        self.model = model
        self.feats = feats
        log(f"变体 {variant}：特征 {dim} 维、序列 {self.seq_len}、训练 {count} 样本")

    def predict_digit(self, t):
        """预测第 t 期（特征只用 [:t]），返回 3 个 0-9 数字。"""
        window = self.feats[max(0, t - self.seq_len):t]
        if len(window) < self.seq_len:
            return None
        probs = self.model.predict(
            window[-self.seq_len:].reshape(1, self.seq_len, -1), verbose=0)[0]
        return np.argmax(probs, axis=-1).astype(int)


def run_variant_backtest(data, variants=('window100', 'sumspan'),
                         eval_len=400, epochs=3, log=print, interrupt=None):
    """对最近 eval_len 期逐期预测，按 evaluation 口径与随机期望对照。"""
    results = {}
    train_end = len(data) - eval_len
    for v in variants:
        strategy = VariantStrategy(data, v, train_end, epochs=epochs, log=log)
        digit_hits = digit_total = exact = 0
        for i, t in enumerate(range(train_end, len(data))):
            if interrupt is not None and interrupt():
                break
            pred = strategy.predict_digit(t)
            if pred is None:
                continue
            actual = data[t]
            digit_hits += int((pred == actual).sum())
            digit_total += 3
            exact += int(''.join(map(str, pred)) == ''.join(map(str, actual)))
        n = digit_total // 3
        results[v] = {
            'variant': v, 'n': n,
            'digit_rate': digit_hits / digit_total if digit_total else 0.0,
            'digit_p': float(binomtest(digit_hits, digit_total, EV.P_DIGIT).pvalue),
            'exact_hits': exact,
            'expected_exact': n * EV.P_EXACT,
            'exact_p': float(binomtest(exact, n, EV.P_EXACT).pvalue),
            'exact_ci': EV.wilson_interval(exact, n),
        }
        log(f"变体 {v}：逐位 {results[v]['digit_rate']:.4f}（基线 0.1000，p={results[v]['digit_p']:.3f}）| "
            f"直选 {exact}/{n}（随机期望 {n * EV.P_EXACT:.1f}，p={results[v]['exact_p']:.3f}）")
    return results


def format_variant_report(results):
    lines = ['模型变体对照（单模型固定训练集，逐期预测最近区间）']
    for v, r in results.items():
        ci = f"[{r['exact_ci'][0]:.5f}, {r['exact_ci'][1]:.5f}]"
        lines.append(
            f"  {v}：逐位命中率 {r['digit_rate']:.4f}（随机基线 0.1000，p={r['digit_p']:.3f}）| "
            f"直选命中 {r['exact_hits']}/{r['n']} vs 随机期望 {r['expected_exact']:.1f}，"
            f"95% CI {ci}，p={r['exact_p']:.3f}")
    lines.append('判读：p 值不显著 = 变体与随机选号无法区分；调模型不改变命中率。')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='排列3命中模拟器')
    parser.add_argument('--mode', default='组选6', choices=list(EV.PLAY_RULES))
    parser.add_argument('--bets', type=int, default=30)
    parser.add_argument('--start', type=int, default=None,
                        help='评估起点（默认 max(60, N//2)）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selection', choices=['random', 'model'], default='random')
    parser.add_argument('--eval-len', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--variants', action='store_true', help='只跑模型变体对照')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    data = EV.load_excel_data()
    log = print
    if args.variants:
        results = run_variant_backtest(data, eval_len=args.eval_len,
                                       epochs=args.epochs, log=log)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(format_variant_report(results))
        return
    if args.selection == 'random':
        start = args.start if args.start is not None else max(60, len(data) // 2)
        result = simulate_random(data, args.mode, args.bets, start=start,
                                 seed=args.seed)
        if args.json:
            result = {k: v for k, v in result.items()
                      if k not in ('cum_pnl', 'win_flags')}
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_simulation_report(result))
        return
    from data_processor import DataProcessor
    from paths import resolve
    dp = DataProcessor(resolve('历史数据.xlsx'), '')
    if not dp.load_data(args.eval_len + MODEL_SIM_HEADROOM, log):
        raise SystemExit('数据加载失败')
    result = simulate_model(dp, args.mode, args.bets, eval_len=args.eval_len,
                            epochs=args.epochs, log=log)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_simulation_report(result))


if __name__ == '__main__':
    main()
