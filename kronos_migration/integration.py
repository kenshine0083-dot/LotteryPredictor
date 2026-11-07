"""把 Kronos 式层级模型接入 LotteryPredictor 的调用约定

现有 lottery_model.LotteryModel 对外暴露 predict_multiple(k, log_callback, mode)，
返回 [{'numbers': '049', 'score': 0.0142}]，score 为该注的模型概率。
本模块提供同签名实现，便于在 GUI / 模拟器里直接替换。

与旧实现的差别：旧模型输出三个独立的 (3,10) 边际分布，再相乘得到组合概率，
隐含「三位条件独立」假设；本模型输出 1000 维联合分布，天然包含位间交互。
"""

import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from kronos_lottery import (CKPT_DIR, KronosLottery, N_FINE, load_sequence,
                            shape_of_index)

COMBO_MAX_BETS = {"直选": 1000, "组选6": 120, "组选3": 90}
COMBO_PER_BET = {"直选": 1, "组选6": 6, "组选3": 3}


def iter_combos(mode):
    """与 lottery_model.iter_combos 保持完全一致的定义。"""
    if mode == "直选":
        for a in range(10):
            for b in range(10):
                for c in range(10):
                    yield f"{a}{b}{c}"
    elif mode in ("组选6", "组选3"):
        for a in range(10):
            for b in range(a, 10):
                for c in range(b, 10):
                    distinct = len({a, b, c})
                    if (mode == "组选6" and distinct == 3) or (
                        mode == "组选3" and distinct == 2
                    ):
                        yield f"{a}{b}{c}"
    else:
        raise ValueError(f"未知玩法：{mode}")


class KronosLotteryPredictor:
    """加载已训练的层级模型，输出下一期的组合概率排序。"""

    def __init__(self, window=128, ckpt_path=None):
        self.window = window
        self.model = KronosLottery(max_len=window)
        path = ckpt_path or os.path.join(CKPT_DIR, "kronos_lottery.pt")
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        self._seq = None

    def attach(self, seq_idx):
        """传入按时间升序的历史号码索引序列（0-999）。"""
        self._seq = np.asarray(seq_idx, dtype=np.int64)

    def next_joint_probs(self):
        """返回下期 1000 维联合概率。"""
        if self._seq is None or len(self._seq) < self.window:
            raise RuntimeError("历史序列不足或未调用 attach()")
        seg = self._seq[-self.window:]
        xc = torch.from_numpy(shape_of_index(seg)[None, :])
        xf = torch.from_numpy(seg[None, :])
        with torch.no_grad():
            lc, lf = self.model(xc, xf)
        pc = F.softmax(lc, dim=-1)[0]
        pf = F.softmax(lf, dim=-1)[0]
        from kronos_lottery import SHAPE_MASK

        joint = (pc.unsqueeze(-1) * pf.unsqueeze(0) * SHAPE_MASK.float()).sum(dim=0)
        return (joint / joint.sum()).numpy()

    def marginal_probs(self):
        """(3, 10) 边际概率，用于和旧模型的输出口径对照。"""
        joint = self.next_joint_probs().reshape(10, 10, 10)
        return np.stack([joint.sum(axis=(1, 2)), joint.sum(axis=(0, 2)), joint.sum(axis=(0, 1))])

    def predict_multiple(self, k, log_callback=print, mode="直选"):
        if mode not in COMBO_MAX_BETS:
            log_callback("玩法必须是 直选 / 组选6 / 组选3")
            return None
        joint = self.next_joint_probs()
        k = int(k)
        if not 1 <= k <= COMBO_MAX_BETS[mode]:
            log_callback(f"{mode} 注数必须是 1~{COMBO_MAX_BETS[mode]} 的整数")
            return None

        combos = []
        for s in iter_combos(mode):
            digits = (int(s[0]), int(s[1]), int(s[2]))
            if mode == "直选":
                score = float(joint[digits[0] * 100 + digits[1] * 10 + digits[2]])
            else:
                score = float(sum(
                    joint[p[0] * 100 + p[1] * 10 + p[2]]
                    for p in set(itertools.permutations(digits))
                ))
            combos.append({"numbers": s, "score": score})
        combos.sort(key=lambda x: (-x["score"], x["numbers"]))
        top = combos[:k]
        per_bet = COMBO_PER_BET[mode]
        log_callback(
            f"[Kronos层级模型] {k} 注不重复{mode}组合；"
            f"单期中奖概率 {k * per_bet / 1000:.2%}（与随机选注相同）"
        )
        return top


def _demo():
    _, nums, seq_idx = load_sequence()
    predictor = KronosLotteryPredictor()
    predictor.attach(seq_idx)
    print(f"历史 {len(seq_idx)} 期，最新一期 {nums[-1].tolist()}")
    print(f"模型给出的下期最高概率组合：{predictor.predict_multiple(5, print, '直选')}")
    marg = predictor.marginal_probs()
    print("三位边际概率最高值：", [f"{m.max():.3f}" for m in marg])
    print("（若均匀，每位约 0.100；显著偏离才说明模型看到了结构）")


if __name__ == "__main__":
    _demo()
