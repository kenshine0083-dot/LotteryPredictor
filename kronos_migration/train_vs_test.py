"""训练集 vs 测试集命中率对照

模型在训练集 loss 明显下降（7.10 -> 5.77），但在样本外没有超出随机。
本脚本判定这属于「学到规律」还是「记住样本」：

- 若训练集命中率显著高于 k/1000、测试集等于 k/1000  -> 纯记忆（过拟合）
- 若两者都等于 k/1000                                -> 数据无可学结构
- 若训练集和测试集都显著高于 k/1000                   -> 存在真实结构（本例不应出现）
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from kronos_lottery import (CKPT_DIR, KronosLottery, SHAPE_MASK, load_sequence,
                            predict_test, shape_of_index)

WINDOW = 64


def rate_table(probs, y_true, ks=(1, 5, 10, 20)):
    order = np.argsort(-probs, axis=1)
    out = {}
    for k in ks:
        hits = int((order[:, :k] == y_true[:, None]).any(axis=1).sum())
        out[k] = (hits, len(y_true), hits / len(y_true))
    return out


def main():
    _, nums, seq_idx = load_sequence()
    seq_shape = shape_of_index(seq_idx)
    split = int(len(seq_idx) * 0.8)

    model = KronosLottery(max_len=WINDOW)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "kronos_lottery.pt"),
                                     map_location="cpu"))
    model.eval()

    print(f"训练段 {split} 期 / 测试段 {len(seq_idx) - split} 期\n")

    # 训练段：模型在训练时见过这些窗口，命中率高即等于记忆
    probs_tr = predict_test(model, seq_idx, seq_shape, WINDOW, WINDOW)
    y_tr = seq_idx[WINDOW:split]
    probs_tr = probs_tr[: len(y_tr)]

    # 测试段
    probs_te = predict_test(model, seq_idx, seq_shape, WINDOW, split)
    y_te = seq_idx[split:]

    tr = rate_table(probs_tr, y_tr)
    te = rate_table(probs_te, y_te)

    print(f"{'k注':>5}{'训练命中率':>14}{'测试命中率':>14}{'理论':>10}"
          f"{'训练倍数':>12}{'测试倍数':>12}")
    print("-" * 70)
    for k in tr:
        _, n_tr, r_tr = tr[k]
        _, n_te, r_te = te[k]
        base = k / 1000
        print(f"{k:>5}{r_tr:>14.4f}{r_te:>14.4f}{base:>10.4f}"
              f"{r_tr / base:>12.2f}{r_te / base:>12.2f}")

    print()
    # 形态（coarse）预测准确率：基线为 72%（全猜组6）
    shape_mask_np = SHAPE_MASK.numpy().astype(np.float32)   # (3, 1000)

    def shape_dist_of(probs):
        return probs @ shape_mask_np.T                       # (N, 3)

    def shape_acc(probs, y):
        pred = shape_dist_of(probs).argmax(axis=1)
        return (pred == shape_of_index(y)).mean()

    print(f"形态预测准确率（基线=0.720，即全猜组6）：")
    print(f"  训练段 {shape_acc(probs_tr, y_tr):.4f}")
    print(f"  测试段 {shape_acc(probs_te, y_te):.4f}")

    # 补:测试集上形态预测是否显著优于基线
    pred_shape = shape_dist_of(probs_te).argmax(axis=1)
    true_shape = shape_of_index(y_te)
    hits = int((pred_shape == true_shape).sum())
    p_base = 0.72
    print(f"  测试段形态命中 {hits}/{len(y_te)} = {hits / len(y_te):.4f}，"
          f"对基线 0.72 的二项检验 p="
          f"{stats.binomtest(hits, len(y_te), p_base, alternative='greater').pvalue:.4f}")

    print()
    print("判读：若训练段倍数明显 > 1 而测试段 ≈ 1，说明模型记住了训练样本而非学到规律。")


if __name__ == "__main__":
    main()
