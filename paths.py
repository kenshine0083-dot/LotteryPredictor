"""路径解析与配置读取（独立小模块）。

settlement / simulation 的 CLI 只需要路径与配置；此前它们默认路径时
`import main`，会连带导入 TensorFlow（数秒 + 数百 MB）。把 _resolve 与
config 读取抽到这里，CLI 保持轻量。main.py 与测试仍可用 main._resolve
（别名）。
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def resolve(value):
    """把配置里的相对路径解析成相对本文件目录的绝对路径，消除对 CWD 的依赖。"""
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def load_config():
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        return json.load(f)
