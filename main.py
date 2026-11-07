import sys
import tkinter as tk
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_processor import DataProcessor
from gui import LotteryPredictorGUI
from lottery_model import LotteryModel, SEQUENCE_LENGTH_DEFAULT
from paths import load_config, resolve

# 兼容旧调用：子模块与测试通过 main._resolve 取路径。
_resolve = resolve


def main():
    config = load_config()

    data_processor = DataProcessor(resolve(config['default_file_path']), config['fetch_url'])
    lottery_model = LotteryModel(sequence_length=SEQUENCE_LENGTH_DEFAULT,
                                 data_processor=data_processor)

    root = tk.Tk()
    LotteryPredictorGUI(root, data_processor, lottery_model, resolve(config['save_dir']))
    root.mainloop()


if __name__ == "__main__":
    main()
