"""累积审计可视化：命中率收敛曲线 + 统计功效曲线

左图回答「模型选号长期表现如何」：把所有存档按时间累积，画出累计命中率
如何随样本增加而变化，并与理论值 k/1000 对照。
右图回答「要证明模型有优势需要多少样本」：这是判断任何「命中率超常」
现象是否可信的尺子。

注：绘图依赖 matplotlib，项目运行环境（py3.8 + TF）不含该库，
     用 C:/PythonProject/Kronos/.venv 运行本脚本即可。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit
import evaluation as EV

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
MODE = "组选6"


def cumulative_series(reports, mode):
    """把该玩法所有「存档 × 期」事件按时序累积，返回累计注数/命中/命中率。"""
    events = []
    for r in reports:
        if r["mode"] != mode:
            continue
        for d in r["details"]:
            events.append((d["period"], r["bets"], int(d["win_bets"])))
    events.sort(key=lambda x: x[0])
    cum_bets = np.cumsum([e[1] for e in events])
    cum_wins = np.cumsum([e[2] for e in events])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = cum_wins / cum_bets
    return cum_bets, cum_wins, rate, len(events)


def main():
    reports, errors = audit.collect()
    if not reports:
        print("暂无存档可审计，请先运行程序生成选号并更新开奖数据")
        return

    cum_bets, cum_wins, rate, n_events = cumulative_series(reports, MODE)
    p0 = EV.PLAY_RULES[MODE]["hit_per_bet"]
    summary = audit.summarize(reports).get(MODE)
    if summary is None or n_events == 0:
        print(f"暂无可核对的 {MODE} 开奖记录")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    # ---- 左：累计命中率收敛 ----
    ax1.plot(cum_bets, rate, color="#1f4e79", linewidth=1.8, label="累计实际命中率")
    ax1.axhline(p0, color="#c00000", linestyle="--", linewidth=1.5,
                label=f"理论值 {p0:.4f}")
    idx = np.unique(np.linspace(0, len(cum_bets) - 1, 160).astype(int))
    lo, hi = [], []
    for i in idx:
        a, b = EV.wilson_interval(int(cum_wins[i]), int(cum_bets[i]))
        lo.append(a)
        hi.append(b)
    ax1.fill_between(cum_bets[idx], lo, hi, color="#1f4e79", alpha=0.13,
                     label="Wilson 95% 置信区间")
    ax1.set_xlabel("累计投注注数")
    ax1.set_ylabel("命中率")
    ax1.set_title(f"{MODE}：累计命中率随样本收敛到理论值", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.grid(alpha=0.25)
    ax1.set_ylim(0, max(p0 * 4, float(np.nanmax(rate[: max(5, len(rate) // 2)])) * 1.05))

    final_txt = (f"累计 {int(cum_bets[-1])} 注 / 中 {int(cum_wins[-1])} 注\n"
                 f"实际 {rate[-1]:.6f}  vs  理论 {p0:.6f}\n"
                 f"倍数 {rate[-1] / p0:.2f}x   p={summary['p_value']:.3f}")
    ax1.text(0.03, 0.05, final_txt, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                       edgecolor="#cccccc"))

    # ---- 右：统计功效 ----
    lifts = np.linspace(0.03, 1.2, 120)
    needs = np.array([audit.required_samples(p0, l) for l in lifts])
    ax2.plot(lifts * 100, needs, color="#2e7d32", linewidth=1.8)
    ax2.axhline(cum_bets[-1], color="#c00000", linestyle="--", linewidth=1.3,
                label=f"当前累计 {int(cum_bets[-1])} 注")
    for lift in (0.10, 0.50, 1.00):
        need = audit.required_samples(p0, lift)
        ax2.plot(lift * 100, need, "o", color="#2e7d32", markersize=5)
        ax2.annotate(f"+{lift * 100:.0f}% 需 {need:,.0f} 注",
                     xy=(lift * 100, need), xytext=(6, 8),
                     textcoords="offset points", fontsize=9, color="#2e7d32")
    ax2.set_yscale("log")
    ax2.set_xlabel("需要检测出的真实优势幅度（相对提升 %）")
    ax2.set_ylabel("所需投注注数（对数刻度）")
    ax2.set_title(f"统计功效：证明「{MODE}更准」需要的样本量（基线 p={p0:.4f}）",
                  fontsize=12)
    ax2.legend(loc="upper right", fontsize=9.5)
    ax2.grid(alpha=0.25, which="both")

    fig.tight_layout()
    out = os.path.join(BASE, "audit_result.png")
    fig.savefig(out, dpi=140)
    print(f"已保存: {out}")
    if errors:
        print(f"（{len(errors)} 份存档读取失败）")


if __name__ == "__main__":
    main()
