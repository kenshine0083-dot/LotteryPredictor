"""玩法组合的纯计算层：枚举、排序、滑窗，不依赖 TensorFlow / keras-tuner。

从 ``lottery_model.py`` 抽出的这几个常量与函数与神经网络无关，但被
``simulation.py``、GUI 与轻量入口共用。放在这里之后，模拟器与轻量版 GUI
不再需要从 ``lottery_model`` 顶层导入（那会连带加载 TensorFlow）。

所有数字与 ``evaluation.PLAY_RULES`` 保持一致；这里是「组合空间」的
唯一实现，避免多处各写一份导致枚举口径漂移。
"""
from __future__ import annotations

import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 三个位置各预测 0-9，覆盖全部 1000 种有序且可重复的开奖结果。
OUTPUT_SHAPE = (3, 10)
# 默认输入序列长度（config 无该项时的回退值；GUI 与 CLI 同值）。
SEQUENCE_LENGTH_DEFAULT = 50
# 各玩法的组合空间与单注中奖概率（与 evaluation.PLAY_RULES 的数字一致）。
COMBO_MAX_BETS = {'直选': 1000, '组选6': 120, '组选3': 90}
COMBO_PER_BET = {'直选': 1, '组选6': 6, '组选3': 3}


def sliding_windows(matrix, window):
    """(N, D) → (N-window+1, window, D) 滑窗数组。

    sliding_window_view 是零拷贝视图，ascontiguousarray 只做一次内存复制，
    等价并取代逐窗列表切片复制（训练集上万窗口时明显更快）。
    """
    view = sliding_window_view(matrix, (window, matrix.shape[1]))[:, 0]
    return np.ascontiguousarray(view)


def iter_combos(mode):
    """按玩法枚举全部投注组合字符串（rank_combos / 模拟器抽样 / 全收共用）。

    直选：1000 个有序三位组合；组选6：三个互不相同数字的无序组（120 个）；
    组选3：恰有一对相同数字的无序组（90 个）。未知玩法抛 ValueError。
    """
    if mode == '直选':
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    yield f'{a}{b}{c}'
    elif mode in ('组选6', '组选3'):
        for a in range(10):
            for b in range(a, 10):
                for c in range(b, 10):
                    distinct = len({a, b, c})
                    if (mode == '组选6' and distinct == 3) or (mode == '组选3' and distinct == 2):
                        yield f'{a}{b}{c}'
    else:
        raise ValueError(f'未知玩法：{mode}')


def rank_combos(probs, mode, k):
    """按模型概率降序给出 k 注互不重复组合（predict_multiple 与模拟器共用）。

    probs 形状 (3, 10)；组选按无序数字组枚举，score 为该组任一排列
    的概率之和，即该注的中奖概率（模型口径）。注意 score 只决定「选哪 k 注」，
    不是「这 k 注更容易中奖」—— 真实概率恒为玩法覆盖率。
    """
    if mode not in COMBO_MAX_BETS:
        raise ValueError(f'未知玩法：{mode}')
    k = int(k)
    if not 1 <= k <= COMBO_MAX_BETS[mode]:
        raise ValueError(f'{mode} 注数须在 1~{COMBO_MAX_BETS[mode]} 之间')

    def ordered_score(x, y, z):
        return float(probs[0, x] * probs[1, y] * probs[2, z])

    combos = []
    for s in iter_combos(mode):
        digits = (int(s[0]), int(s[1]), int(s[2]))
        if mode == '直选':
            score = ordered_score(*digits)
        else:
            score = sum(ordered_score(*p) for p in set(itertools.permutations(digits)))
        combos.append((float(score), s))
    combos.sort(key=lambda item: (-item[0], item[1]))
    return [{'numbers': num, 'score': score} for score, num in combos[:k]]
