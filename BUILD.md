# 构建 Windows GUI 可执行程序

项目本体是 Python 脚本，仓库**不分发二进制文件**。本文件说明如何在自己机器上
生成 exe，以及两种模式各自的能力边界。

## 前置条件

| 模式 | 入口 | 需要的环境 |
|---|---|---|
| `lite`（推荐） | `main_lite.py` | Python 3.8–3.13 + `tkinter` + `requirements-core.txt` + `scikit-learn`、`requests`、`tenacity`、`PyInstaller` |
| `full` | `main.py` | `requirements.txt` 全量，含 **TensorFlow 2.13**（仅 Python 3.8–3.11）+ `PyInstaller` |

先装打包工具：

```bash
pip install pyinstaller
```

检查环境是否满足（不构建、不产生文件）：

```bash
python build_exe.py --check --mode lite
python build_exe.py --check --mode full
```

## 构建

```bash
python build_exe.py --mode lite              # 无控制台窗口
python build_exe.py --mode lite --console    # 保留控制台，便于看启动报错
python build_exe.py --mode full              # 完整版，体积大、耗时长
python build_exe.py --mode lite --clean      # 清理旧的 dist/ build/ *.spec 后重建
```

产物：

```
dist/LotteryPredictorLite/LotteryPredictorLite.exe      # lite
dist/LotteryPredictor/LotteryPredictor.exe              # full
```

脚本会把 `历史数据.xlsx`、`config.json`、`requirements.txt` 一并放进产物目录，
并用 `--onedir` 而非 `--onefile`（后者每次启动都要解压数百 MB，体验更差）。
`dist/`、`build/`、`*.spec` 都在 `.gitignore` 里，**不要提交**。

## 两种模式的实际差别

**`lite`** 用 `LiteModel` 替换神经网络层：

- 选号改为从玩法组合空间**均匀随机抽取**。这不是功能缩水 —— 9 项检验（含 Holm
  校正）无法拒绝「独立均匀随机」，任何选号策略的单期中奖概率都等于玩法覆盖率，
  模型打分本来就不改变命中率。
- 结果区标签显示「单注理论概率」而不是「模型打分」，`score` 是真实覆盖率。
- 随机性检验、账目与期望值、跨存档累积审计、命中率回测（随机基线）、数据更新
  与对账全部保留。
- 「初次运行 / 继续运行」只加载数据并做统计检验，不训练、不读写模型权重。

为了让 lite 真的不带 TensorFlow，组合枚举等纯计算已抽到 `combos.py`；
`simulation.py` 不再从 `lottery_model` 顶层导入。构建时仍显式排除
`tensorflow` / `keras` / `keras_tuner` / `lottery_model` 等模块，因为模拟器里的
`--with-ml` 分支仍有延迟导入，PyInstaller 的静态分析会顺藤摸瓜。
**因此 lite 版不支持需要 TF 的 ML 回放分支**；这是有意的取舍，不是缺陷。

**`full`** 包含模型训练与权重加载，功能与 `python main.py` 完全一致，
但产物通常数百 MB 起，且只能在 TF 2.13 支持的 Python 版本上构建。

## 已知限制

- 只针对 Windows 验证过；macOS / Linux 需要调整 `--add-data` 的分隔符（`:`）。
- `--windowed` 下看不到 stdout，启动即闪退时用 `--console` 重建排查。
- exe 是本地便利产物，**不受 CI 覆盖**；CI 只验证纯统计链路与文档数字。
- 打包不会把 `picks/`、`saved_state/` 等私有目录带进去；它们也从未被 git 跟踪。
- 首次运行会联网更新开奖数据（体彩公开接口），离线环境可先用仓库自带工作簿。

## 验证记录（本机）

构建脚本的 `--check` 只做环境校验，CI 会执行它；实际构建与产物体积请在自己
机器上按上面命令完成后记录，不要写进仓库文档（不同机器差异很大）。
