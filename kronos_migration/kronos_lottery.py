"""Kronos 思想迁移到排列3：层级 token + 自回归

## 迁移映射

Kronos 的两级量化是这样工作的：
    连续 OHLCV 向量 --(分层向量量化)--> s1 token（粗，结构） + s2 token（细，残差细节）

排列3 的号码天然是离散的，但它同样有「粗结构 + 细排列」的两层：
    3 位号码 --> coarse token（形态：组6/组3/豹子） + fine token（该形态下的具体排列）

于是联合分布被分解为：
    P(combo | history) = P(coarse | history) x P(fine | coarse, history)

这正是 Kronos 先 coarse 再 coarse-conditioned fine 的作法。区别在于
Kronos 的量化器需要学习，而彩票的层级划分是规则式的，不需要额外训练。

## 模型

decoder-only Transformer（因果掩码），输入是历史各期的 (coarse, fine) 双层
嵌入之和加位置编码，在最后位置输出两个分类头，各自预测下一期的 coarse 与 fine。

## 目的

本脚本用于严格检验「该结构能否带来超出随机的命中率」。评估采用时间切分的
样本外滚动预测，并以二项检验对比理论命中率 k/1000。
"""

import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "历史数据.xlsx")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# 形态编码：0=组6(三位互异) 1=组3(一对) 2=豹子
N_COARSE = 3
N_FINE = 1000

# 每种形态下合法的细 token 集合
SHAPE_MEMBERS = {0: [], 1: [], 2: []}
for _combo in itertools.product(range(10), repeat=3):
    _idx = _combo[0] * 100 + _combo[1] * 10 + _combo[2]
    _d = len(set(_combo))
    SHAPE_MEMBERS[{3: 0, 2: 1, 1: 2}[_d]].append(_idx)

# 每个细 token 所属形态，用于损失掩码与联合概率
FINE_TO_SHAPE = np.zeros(N_FINE, dtype=np.int64)
for _s, _members in SHAPE_MEMBERS.items():
    for _m in _members:
        FINE_TO_SHAPE[_m] = _s

# 联合概率归一化用的布尔掩码：(3, 1000)
SHAPE_MASK = torch.zeros(N_COARSE, N_FINE, dtype=torch.bool)
for _s, _members in SHAPE_MEMBERS.items():
    SHAPE_MASK[_s, _members] = True


def shape_of_index(idx_arr):
    """无符号整数号码 -> 形态编号。

    distinct_pairs 统计三位之间"不同的对数"：
        全不同 = 3 对  -> 组6 (0)
        一对相同 = 2 对 -> 组3 (1)
        三位全同 = 0 对 -> 豹子 (2)
    """
    a = idx_arr // 100
    b = (idx_arr // 10) % 10
    c = idx_arr % 10
    distinct_pairs = (
        (a != b).astype(np.int64) + (b != c).astype(np.int64) + (a != c).astype(np.int64)
    )
    return np.where(distinct_pairs == 3, 0, np.where(distinct_pairs == 2, 1, 2))


def load_sequence():
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    df["Period"] = df["Period"].astype(int)
    df = df.sort_values("Period").reset_index(drop=True)
    nums = np.array([[int(t) for t in str(r).split()] for r in df["Numbers"]], dtype=np.int64)
    idx = nums[:, 0] * 100 + nums[:, 1] * 10 + nums[:, 2]
    return df, nums, idx


class CausalBlock(nn.Module):
    """预归一化的因果自注意力块。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class KronosLottery(nn.Module):
    """双层 token 的自回归序列模型。"""

    def __init__(self, d_model=96, n_layers=3, n_heads=4, max_len=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.coarse_emb = nn.Embedding(N_COARSE, d_model)
        self.fine_emb = nn.Embedding(N_FINE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head_coarse = nn.Linear(d_model, N_COARSE)
        self.head_fine = nn.Linear(d_model, N_FINE)

        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, coarse_ids, fine_ids):
        b, l = coarse_ids.shape
        pos = torch.arange(l, device=coarse_ids.device).unsqueeze(0).expand(b, l)
        x = self.coarse_emb(coarse_ids) + self.fine_emb(fine_ids) + self.pos_emb(pos)
        x = self.drop(x)
        mask = self.causal_mask[:l, :l]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x[:, -1, :])
        return self.head_coarse(x), self.head_fine(x)


def build_windows(seq_idx, seq_shape, window):
    """滑窗：返回 (N, window) 的输入与 (N,) 的下期标签。"""
    n = len(seq_idx)
    count = n - window
    x_coarse = np.stack([seq_shape[i:i + window] for i in range(count)])
    x_fine = np.stack([seq_idx[i:i + window] for i in range(count)])
    y_coarse = seq_shape[window:window + count]
    y_fine = seq_idx[window:window + count]
    return x_coarse, x_fine, y_coarse, y_fine


def train_model(seq_idx, seq_shape, window=64, epochs=15, batch_size=128,
                lr=6e-4, seed=0, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 只用训练段构造样本，避免验证期信息进入参数
    split = int(len(seq_idx) * 0.8)
    tr_idx, tr_shape = seq_idx[:split], seq_shape[:split]
    xc, xf, yc, yf = build_windows(tr_idx, tr_shape, window)

    model = KronosLottery(max_len=window)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xc)
    log(f"训练样本 {n} 条，窗口 {window}，参数量 "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        total, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            sel = perm[i:i + batch_size]
            lc, lf = model(torch.from_numpy(xc[sel]), torch.from_numpy(xf[sel]))
            loss_c = F.cross_entropy(lc, torch.from_numpy(yc[sel]))
            # 细粒度头只在真实形态对应的候选集内比较，与 Kronos 的层级解码一致
            mask = SHAPE_MASK[torch.from_numpy(yc[sel])]
            lf = lf.masked_fill(~mask, float("-inf"))
            loss_f = F.cross_entropy(lf, torch.from_numpy(yf[sel]))
            loss = loss_c + loss_f
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(sel)
            cnt += len(sel)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  epoch {ep + 1:3d}/{epochs}  loss={total / cnt:.4f}")
    return model


@torch.no_grad()
def predict_test(model, seq_idx, seq_shape, window, test_start, batch_size=128):
    """对测试段每一期做样本外预测，返回每个细 token 的联合概率矩阵 (N_test, 1000)。"""
    model.eval()
    n = len(seq_idx)
    starts = list(range(test_start, n))
    out = []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        xc = np.stack([seq_shape[t - window:t] for t in chunk])
        xf = np.stack([seq_idx[t - window:t] for t in chunk])
        lc, lf = model(torch.from_numpy(xc), torch.from_numpy(xf))
        pc = F.softmax(lc, dim=-1)                       # (B, 3)
        pf = F.softmax(lf, dim=-1)                       # (B, 1000)
        joint = pc.unsqueeze(-1) * pf.unsqueeze(1)       # (B, 3, 1000)
        joint = joint * SHAPE_MASK.unsqueeze(0)
        # 每个号码只属于一个形态，对形态维求和即得该号码的联合概率 (B, 1000)
        flat = joint.sum(dim=1)
        total = flat.sum(dim=1, keepdim=True)
        out.append((flat / total).numpy())
    return np.concatenate(out, axis=0)


def evaluate(probs, y_true, ks=(1, 3, 5, 10, 20)):
    """按 top-k 命中率评估，并对每个 k 做二项检验。"""
    n_test = len(y_true)
    order = np.argsort(-probs, axis=1)
    rows = []
    for k in ks:
        topk = order[:, :k]
        hits = int((topk == y_true[:, None]).any(axis=1).sum())
        base = k / 1000
        pval = stats.binomtest(hits, n_test, base, alternative="greater").pvalue
        rows.append({
            "k": k,
            "hits": hits,
            "n_test": n_test,
            "hit_rate": hits / n_test,
            "baseline": base,
            "lift": (hits / n_test) / base,
            "p_value": pval,
        })
    return rows


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    df, nums, seq_idx = load_sequence()
    seq_shape = shape_of_index(seq_idx)
    window = 64
    split = int(len(seq_idx) * 0.8)

    print(f"数据 {len(seq_idx)} 期，训练段 {split} 期，测试段 {len(seq_idx) - split} 期")
    print(f"层级 token：coarse={N_COARSE} 类（组6/组3/豹子），fine={N_FINE} 类")
    print()

    t0 = time.time()
    ckpt = os.path.join(CKPT_DIR, "kronos_lottery.pt")
    if "--eval-only" in sys.argv and os.path.exists(ckpt):
        model = KronosLottery(max_len=window)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"已加载既有权重 {ckpt}，跳过训练")
    else:
        model = train_model(seq_idx, seq_shape, window=window, epochs=15)
        print(f"训练耗时 {time.time() - t0:.1f}s\n")
        torch.save(model.state_dict(), ckpt)

    print("样本外滚动预测 ...")
    t0 = time.time()
    probs = predict_test(model, seq_idx, seq_shape, window, split)
    y_true = seq_idx[split:]
    print(f"完成，耗时 {time.time() - t0:.1f}s\n")

    rows = evaluate(probs, y_true)
    print(f"{'k注':>4}{'命中':>8}{'命中率':>10}{'理论':>10}{'倍数':>8}{'p值':>10}")
    print("-" * 52)
    for r in rows:
        print(f"{r['k']:>4}{r['hits']:>8}{r['hit_rate']:>10.4f}"
              f"{r['baseline']:>10.4f}{r['lift']:>8.2f}{r['p_value']:>10.4f}")

    np.save(os.path.join(CKPT_DIR, "test_probs.npy"), probs)
    np.save(os.path.join(CKPT_DIR, "test_true.npy"), y_true)

    print()
    print("参考：随机选 k 注的理论命中率为 k/1000；p<0.05 才说明模型显著优于随机。")


if __name__ == "__main__":
    main()
