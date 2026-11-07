"""轻量 GUI 入口：不加载 TensorFlow / keras-tuner，只保留统计与选号功能。

完整版入口是 ``main.py``（带神经网络训练）；本入口用一个与 ``LotteryModel``
同接口的 ``LiteModel`` 替换模型层：

- 选号改为从玩法组合空间**均匀随机抽取**（本来就该如此：数据经九项检验与
  独立均匀随机相容，任何选号策略的单期中奖概率都等于玩法覆盖率）；
- ``score`` 字段显示的是**单注理论中奖概率**，不是模型打分，更不是排名依据；
- 初次运行只加载数据并做统计检验，不做训练；模型相关按钮给出明确提示。

这样打包出来的 exe 不需要数百 MB 的 TensorFlow，能在只装
numpy / pandas / scipy / openpyxl / scikit-learn / tkinter 的环境里运行。
"""
from __future__ import annotations

import random
import sys
import tkinter as tk
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from combos import COMBO_MAX_BETS, COMBO_PER_BET, iter_combos
from data_processor import DataProcessor
from gui import LotteryPredictorGUI
from paths import load_config, resolve


class LiteModel:
    """与 LotteryModel 同接口的均匀随机选号器（无神经网络、无权重文件）。"""

    # 非 None 哨兵：GUI 用 `model is None` 判断「尚未加载模型」；
    # 轻量版永远不需要加载模型，故用一个标记对象让流程正常走通。
    model = object()
    score_label = '单注理论概率'

    def __init__(self, data_processor=None):
        self.data_processor = data_processor
        self.rng = random.Random()      # 系统熵播种，不固定，避免可预测序列

    def predict_multiple(self, k, log_callback, mode='直选'):
        if mode not in COMBO_MAX_BETS:
            log_callback('玩法必须是 直选 / 组选6 / 组选3')
            return None
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 0
        max_bets = COMBO_MAX_BETS[mode]
        if not 1 <= k <= max_bets:
            log_callback(f'{mode} 注数必须是 1~{max_bets} 的整数')
            return None

        pool = list(iter_combos(mode))
        chosen = self.rng.sample(pool, k)
        per_bet = COMBO_PER_BET[mode] / 1000
        log_callback(
            f'轻量版：从 {mode} 的 {len(pool)} 个组合中均匀随机抽取 {k} 注'
            f'（不含模型打分）；单期中奖概率 {k * per_bet:.2%}，与选号方式无关')
        return [{'numbers': c, 'score': per_bet} for c in chosen]

    def train(self, log_callback, interrupt_flag, progress_callback=None):
        log_callback('轻量版不含神经网络训练：仅加载历史数据并做统计检验。'
                     '数据随机性结论与完整版一致。')
        if progress_callback:
            progress_callback(1.0)
        return True

    def load_state(self, save_dir, log_callback):
        log_callback('轻量版无模型权重，跳过模型状态加载。')
        return True

    def save_state(self, save_dir, log_callback):
        log_callback('轻量版无模型权重，无需保存模型状态。')
        return True


def main():
    config = load_config()
    data_processor = DataProcessor(resolve(config['default_file_path']),
                                   config['fetch_url'])
    model = LiteModel(data_processor)

    root = tk.Tk()
    app = LotteryPredictorGUI(root, data_processor, model,
                              resolve(config['save_dir']))
    root.title(f'排列3 分析与选号工具 {app_version()} · 轻量版（不含模型）')
    root.mainloop()


def app_version():
    import gui
    return gui.APP_VERSION


if __name__ == '__main__':
    main()
