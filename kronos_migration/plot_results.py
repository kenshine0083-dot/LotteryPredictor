"""生成迁移实验的两张核心图表"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(OUT, "checkpoints")

KS = [1, 5, 10, 20]
TRAIN = [0.1024, 0.2669, 0.3643, 0.4777]
TEST = [0.0006, 0.0032, 0.0084, 0.0174]
THEORY = [k / 1000 for k in KS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})

# ---- 左图：训练 vs 测试命中率 ----
x = np.arange(len(KS))
w = 0.27
b1 = ax1.bar(x - w, TRAIN, w, label="训练集（模型背过）", color="#c00000")
b2 = ax1.bar(x, TEST, w, label="测试集（样本外）", color="#1f4e79")
b3 = ax1.bar(x + w, THEORY, w, label="随机基线 k/1000", color="#a6a6a6")
for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width() / 2, h * 1.15, f"{h * 100:.2f}%",
                 ha="center", va="bottom", fontsize=8.5)
ax1.set_yscale("log")
ax1.set_ylim(5e-4, 1.2)
ax1.set_xticks(x)
ax1.set_xticklabels([f"k={k}" for k in KS])
ax1.set_ylabel("命中率（对数刻度）")
ax1.set_title("训练集命中率 102 倍于随机，测试集跌破随机基线", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(axis="y", alpha=0.25, which="both")

# ---- 右图：随机性检验 p 值 ----
labels = ["第1位均匀", "第2位均匀", "第3位均匀", "三位独立", "期际独立·位1",
          "期际独立·位2", "期际独立·位3", "重复号形态", "和值分布"]
pvals = [0.6675, 0.8249, 0.1495, 0.3743, 0.4791, 0.9801, 0.1222, 0.3106, 0.2064]
y = np.arange(len(labels))[::-1]
colors = ["#2e7d32" if p > 0.05 else "#c00000" for p in pvals]
ax2.barh(y, pvals, color=colors, height=0.6)
ax2.axvline(0.05, color="#c00000", linestyle="--", linewidth=1.3)
ax2.text(0.055, len(labels) - 0.6, "显著性阈值 0.05", color="#c00000", fontsize=9)
for yi, p in zip(y, pvals):
    ax2.text(p + 0.015, yi, f"{p:.3f}", va="center", fontsize=9)
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=9.5)
ax2.set_xlim(0, 1.12)
ax2.set_xlabel("p 值")
ax2.set_title("7,770 期数据 · 9 项随机性检验全部不显著", fontsize=12)
ax2.grid(axis="x", alpha=0.25)

fig.tight_layout()
path = os.path.join(OUT, "kronos_migration_result.png")
fig.savefig(path, dpi=140)
print(f"已保存: {path}")
