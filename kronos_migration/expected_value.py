"""排列3 期望值与返奖率计算

模型能不能提高命中率是一回事，这个游戏本身值不值得玩是另一回事。
本脚本按官方玩法与奖金，计算不同投注方式下的期望收益。
"""

import os

STAKE = 2.0  # 单注 2 元

PLAYS = {
    "直选": {"payout": 1040.0, "hit_per_bet": 1 / 1000, "max_bets": 1000},
    "组选3": {"payout": 346.0, "hit_per_bet": 3 / 1000, "max_bets": 90},
    "组选6": {"payout": 173.0, "hit_per_bet": 6 / 1000, "max_bets": 120},
}


def main():
    print("排列3 单注 2 元，各玩法期望收益")
    print("=" * 66)
    print(f"{'玩法':<8}{'单注中奖率':>12}{'奖金':>10}{'期望回收':>12}{'返奖率':>10}")
    print("-" * 66)
    for name, cfg in PLAYS.items():
        ev = cfg["payout"] * cfg["hit_per_bet"]
        rate = ev / STAKE
        print(f"{name:<8}{cfg['hit_per_bet'] * 100:>11.2f}%{cfg['payout']:>10.0f}"
              f"{ev:>12.3f}{rate * 100:>9.1f}%")

    print()
    print("买满该玩法全部组合的情况（注意：组选3/组选6 只覆盖对应形态，并非必中）")
    print("-" * 66)
    for name, cfg in PLAYS.items():
        cost = cfg["max_bets"] * STAKE
        # 买满 coverage 只等于该玩法覆盖的号码空间，直选才是 100%
        p_cover = min(1.0, cfg["max_bets"] * cfg["hit_per_bet"])
        ev_prize = p_cover * cfg["payout"]
        print(f"{name:<8} 成本 {cost:>6.0f} 元  中奖概率 {p_cover * 100:>5.1f}%  "
              f"期望奖金 {ev_prize:>7.2f} 元  期望净亏 {cost - ev_prize:>6.2f} 元")

    print()
    print("买 k 注（k 注互不重复）的期望值")
    print("-" * 66)
    print(f"{'玩法':<8}{'k':>4}{'成本':>8}{'中奖概率':>10}{'期望奖金':>10}{'期望净收益':>12}")
    for name, cfg in PLAYS.items():
        for k in (1, 5, 10, 50):
            if k > cfg["max_bets"]:
                continue
            cost = k * STAKE
            p_hit = k * cfg["hit_per_bet"]
            ev_prize = p_hit * cfg["payout"]
            print(f"{name:<8}{k:>4}{cost:>8.0f}{p_hit * 100:>9.2f}%"
                  f"{ev_prize:>10.2f}{ev_prize - cost:>12.2f}")

    print()
    print("=" * 66)
    print("要点：所有玩法期望净收益均为负，且与选号方式无关。")
    print("      提高 k 只会同比例放大成本与中奖概率，期望净收益同步线性变差。")
    print("      任何预测模型都无法改变这一点——除非它能突破 k/1000 的命中率上限。")


if __name__ == "__main__":
    main()
