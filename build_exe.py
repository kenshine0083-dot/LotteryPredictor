"""把 GUI 构建成 Windows 可执行程序（PyInstaller）。

    python build_exe.py --check                 # 只检查环境与入口，不构建
    python build_exe.py --mode lite             # 轻量版（推荐，无需 TensorFlow）
    python build_exe.py --mode full             # 完整版（需要 TF 2.13 环境）
    python build_exe.py --mode lite --console  # 带控制台，便于排查启动错误

两种模式的区别（详见 BUILD.md）：
- ``lite`` 入口 ``main_lite.py``：均匀随机选号 + 全部统计/审计/回测功能，
  不含神经网络；显式排除 tensorflow / keras / keras_tuner / lottery_model。
- ``full`` 入口 ``main.py``：完整功能，包含模型训练与加载，产物数百 MB 起。

产物落在 ``dist/``，中间文件在 ``build/``，两者都在 .gitignore 中，
**二进制不会、也不应该提交进仓库**。
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_FILES = ('历史数据.xlsx', 'config.json', 'requirements.txt')
ENTRIES = {'lite': 'main_lite.py', 'full': 'main.py'}
EXCLUDE_LITE = ('tensorflow', 'keras', 'keras_tuner', 'lottery_model',
                'torch', 'matplotlib', 'IPython', 'notebook')


def check(mode):
    """返回 (是否通过, 逐项说明)。不实际构建。"""
    items = []

    def add(name, ok, detail=''):
        items.append((name, bool(ok), detail))

    add('PyInstaller 可用', importlib.util.find_spec('PyInstaller') is not None)
    add('tkinter 可用', importlib.util.find_spec('tkinter') is not None)
    add('入口存在', (BASE / ENTRIES[mode]).is_file(), ENTRIES[mode])
    for f in DATA_FILES:
        add(f'数据文件 {f}', (BASE / f).is_file())
    if mode == 'lite':
        # 轻量版必须真的不碰 TF：这些模块只是从源码里解耦，不要求环境里没有
        add('combos 纯计算层存在', (BASE / 'combos.py').is_file())
        sim = (BASE / 'simulation.py').read_text(encoding='utf-8')
        tree = ast.parse(sim, filename='simulation.py')
        roots = set()
        for node in tree.body:            # 只看模块顶层，函数内延迟导入不算
            if isinstance(node, ast.Import):
                roots.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split('.')[0])
        add('simulation 不再顶层依赖 lottery_model',
            'lottery_model' not in roots, ', '.join(sorted(roots)))
    else:
        add('完整版需 TensorFlow',
            importlib.util.find_spec('tensorflow') is not None,
            '缺少 TF 时会给出提示，构建仍可尝试')
    return all(ok for _, ok, _ in items), items


def build(mode, console=False, clean=False):
    ok, items = check(mode)
    for name, passed, detail in items:
        print(f"[{'OK' if passed else 'WARN'}] {name}" + (f' — {detail}' if detail else ''))
    if not ok:
        print('环境检查未通过，终止构建。', file=sys.stderr)
        return 2

    dist = BASE / 'dist'
    work = BASE / 'build'
    if clean:
        for d in (dist, work):
            if d.is_dir():
                shutil.rmtree(d)
        for f in BASE.glob('*.spec'):
            f.unlink()

    name = 'LotteryPredictorLite' if mode == 'lite' else 'LotteryPredictor'
    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--onedir',
           '--name', name,
           '--console' if console else '--windowed',
           '--distpath', str(dist), '--workpath', str(work),
           '--specpath', str(BASE)]
    for f in DATA_FILES:
        cmd += ['--add-data', f'{BASE / f};.']
    if mode == 'lite':
        for m in EXCLUDE_LITE:
            cmd += ['--exclude-module', m]
    cmd.append(str(BASE / ENTRIES[mode]))

    print('构建命令：')
    print('  ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
    proc = subprocess.run(cmd, cwd=str(BASE))
    if proc.returncode != 0:
        print(f'PyInstaller 返回 {proc.returncode}，构建失败。', file=sys.stderr)
        return proc.returncode

    exe = dist / name / f'{name}.exe'
    if exe.is_file():
        size = sum(f.stat().st_size for f in (dist / name).rglob('*') if f.is_file())
        print(f'构建完成：{exe}')
        print(f'目录体积约 {size / 1024 / 1024:.1f} MiB（含全部依赖与数据文件）')
    else:
        print('未找到预期产物，请检查 PyInstaller 输出。', file=sys.stderr)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description='构建 Windows GUI 可执行程序')
    parser.add_argument('--mode', choices=('lite', 'full'), default='lite')
    parser.add_argument('--console', action='store_true', help='保留控制台窗口')
    parser.add_argument('--clean', action='store_true', help='清理旧的 dist/build/spec')
    parser.add_argument('--check', action='store_true', help='只检查环境，不构建')
    args = parser.parse_args()
    if args.check:
        ok, items = check(args.mode)
        for name, passed, detail in items:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}"
                  + (f' — {detail}' if detail else ''))
        print('环境检查：' + ('通过' if ok else '未通过'))
        return 0 if ok else 1
    return build(args.mode, args.console, args.clean)


if __name__ == '__main__':
    sys.exit(main())
