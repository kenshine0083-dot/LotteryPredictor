"""项目自检钩子：零第三方依赖，pre-commit 与 CI 共用同一入口。

    python pre_commit_checks.py
    python pre_commit_checks.py --json

检查的是「提交前容易出事、但测试覆盖不到」的静态约束：
  1. 私有目录 / 权重 / 凭据没有被 git 跟踪
  2. 文本里没有 GitHub PAT 之类的密钥字面量
  3. .gitignore 仍保留私有目录规则
  4. 核心统计模块顶层不引入重依赖（只解析 AST，不 import）
  5. 纯文本文件均为 UTF-8 无 BOM
  6. 文档与构建资产齐全

为什么不直接用现成的 pre-commit-hooks 替代：那些钩子管格式，不管本项目
「picks/ 不能外发」「核心链路不能沾 TensorFlow」这类语义约束。
标准格式钩子放在 .pre-commit-config.yaml 里，与本脚本互补。
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'dist', 'build',
             'picks', 'saved_state', 'tuner_dir', 'checkpoints', '.pytest_cache'}
BINARY_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.xlsx', '.xls',
                   '.pyc', '.pyo', '.zip', '.gz', '.7z', '.exe', '.dll',
                   '.pkl', '.h5', '.ckpt', '.pt', '.npy'}
TEXT_SUFFIXES = {'.py', '.md', '.yml', '.yaml', '.json', '.txt', '.csv',
                 '.cfg', '.toml', '.gitignore'}
PRIVATE_PREFIXES = ('picks/', 'saved_state/', 'tuner_dir/',
                    'kronos_migration/checkpoints/')
PRIVATE_SUFFIXES = ('.pkl', '.h5', '.ckpt', '.pt', '.token', '.env',
                    '.gh_token', '.pem')
REQUIRED_ASSETS = (
    'README.md', 'README.en.md', 'LICENSE', 'DATA_SOURCES.md',
    'CHANGELOG.md', 'BUILD.md', '.pre-commit-config.yaml',
    '.github/workflows/ci.yml', 'audit_result.png',
    'kronos_migration/kronos_migration_result.png',
    'records_template.csv', 'build_exe.py', 'main_lite.py',
)
CORE_MODULES = ('paths.py', 'evaluation.py', 'randomness.py', 'settlement.py',
                'audit.py', 'export_aggregate.py', 'reproduce.py',
                'audit_external.py')
ALLOWED_ROOTS = {
    'numpy', 'pandas', 'scipy', 'openpyxl',
    '__future__', 'abc', 'argparse', 'ast', 'collections', 'csv', 'datetime',
    'itertools', 'json', 'math', 'os', 'pathlib', 'random', 're', 'shutil',
    'statistics', 'subprocess', 'sys', 'tempfile', 'time', 'typing',
    'warnings',
}
ALLOWED_ROOTS |= {Path(m).stem for m in CORE_MODULES} | {'paths'}
# 密钥前缀用拼接构造，避免本文件自己命中「密钥字面量」检查
SECRET_PREFIXES = ('gh' + 'p_', 'github' + '_pat_', 'gho' + '_', 'ghs' + '_')


def _iter_text_files():
    for p in sorted(BASE.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(BASE).as_posix()
        if any(part in SKIP_DIRS for part in p.relative_to(BASE).parts[:-1]):
            continue
        if rel.startswith(tuple(SKIP_DIRS)):
            continue
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        if p.suffix.lower() in TEXT_SUFFIXES or p.name in ('.gitignore',):
            yield p, rel


def _git_files():
    try:
        r = subprocess.run(['git', '-C', str(BASE), 'ls-files'],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def check_private_tracked():
    files = _git_files()
    if files is None:
        return True, '不在 git 仓库中，跳过'
    bad = [f for f in files
           if f.startswith(PRIVATE_PREFIXES) or f.lower().endswith(PRIVATE_SUFFIXES)]
    return not bad, f'{len(files)} 个被跟踪文件；违规 {bad[:3]}'


def check_secret_literals():
    hits = []
    for p, rel in _iter_text_files():
        if rel == 'pre_commit_checks.py':
            continue
        try:
            text = p.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        for prefix in SECRET_PREFIXES:
            if prefix in text:
                hits.append(f'{rel}: {prefix}...')
    return not hits, f'命中 {hits[:3]}'


def check_gitignore():
    p = BASE / '.gitignore'
    if not p.is_file():
        return False, '.gitignore 缺失'
    text = p.read_text(encoding='utf-8')
    missing = [k for k in ('picks/', 'saved_state/', 'tuner_dir/',
                           'kronos_migration/checkpoints/', 'dist/')
               if k not in text]
    return not missing, f'缺失规则 {missing}'


def check_core_imports():
    offenders = []
    for name in CORE_MODULES:
        p = BASE / name
        if not p.is_file():
            offenders.append(f'{name}: 文件缺失')
            continue
        tree = ast.parse(p.read_text(encoding='utf-8'), filename=name)
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split('.')[0]]
            else:
                continue
            for root in roots:
                if root not in ALLOWED_ROOTS:
                    offenders.append(f'{name}: {root}')
    return not offenders, f'违规 {offenders[:3]}'


def check_encoding():
    bad = []
    for p, rel in _iter_text_files():
        raw = p.read_bytes()
        if raw.startswith(b'\xef\xbb\xbf'):
            bad.append(f'{rel}: BOM')
            continue
        try:
            raw.decode('utf-8')
        except UnicodeDecodeError:
            bad.append(f'{rel}: 非 UTF-8')
    return not bad, f'异常 {bad[:3]}'


def check_assets():
    missing = [a for a in REQUIRED_ASSETS if not (BASE / a).is_file()]
    return not missing, f'缺失 {missing}'


CHECKS = (
    ('私有目录与凭据未被 git 跟踪', check_private_tracked),
    ('源码与文档无密钥字面量', check_secret_literals),
    ('.gitignore 保留私有目录规则', check_gitignore),
    ('核心模块顶层无重依赖', check_core_imports),
    ('纯文本文件 UTF-8 无 BOM', check_encoding),
    ('文档与构建资产齐全', check_assets),
)


def run_checks():
    items = []
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f'{type(e).__name__}: {e}'
        items.append({'name': name, 'ok': bool(ok), 'detail': detail})
    return items


def main():
    parser = argparse.ArgumentParser(description='项目自检（pre-commit / CI 共用）')
    parser.add_argument('--json', action='store_true', help='输出 JSON 结果')
    args = parser.parse_args()
    items = run_checks()
    passed = all(i['ok'] for i in items)
    if args.json:
        print(json.dumps({'passed': passed, 'total': len(items),
                          'passed_count': sum(i['ok'] for i in items),
                          'items': items}, ensure_ascii=False, indent=2))
    else:
        for i in items:
            print(f"[{'PASS' if i['ok'] else 'FAIL'}] {i['name']}"
                  + (f"  — {i['detail']}" if i['detail'] else ''))
        print(f'项目自检：{sum(i["ok"] for i in items)}/{len(items)} 通过')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
