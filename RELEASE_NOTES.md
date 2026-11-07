# v1.0.0 Release Notes 草稿

> 供在 GitHub 上创建 Release 时粘贴使用。发布前请确认 CI 徽章为绿色。

## 这个版本是什么

排列3 的**统计分析与审计工具**，不是预测工具。它给出互不重复的候选组合、
算清每一步的真实中奖概率与期望亏损、检验历史数据是否随机、并长期审计自己的
选号表现。仓库里唯一涉及神经网络的部分是一个**否证实验**：证明这类模型在本
数据集上学不到任何可利用的结构。

## 结论（没有变化）

- 7,770 期真实开奖数据的九项卡方检验全部 p > 0.05；同一检验族用
  Holm-Bonferroni 校正后，显著项数为 **0**。
- 数据与「独立均匀随机」相容，因此 k 注互不重复的单期中奖概率恒为 k/1000，
  与选号方式无关。
- 0.54M 参数的 Kronos 式自回归 Transformer：训练集 top-1 命中率 10.24%，
  样本外 0.06%（随机基线的 0.64 倍）—— 是记忆，不是规律。
- 项目自身 2,860 注累积审计：17 中，命中率 0.005944 vs 理论 0.006000，
  p = 0.548，净亏 2,779 元，实际返奖率 51.4%。

## v1.0.0 新增

- `randomness.py` 命令行参数化（`--digits` / `--base` / `--data` / `--profile` / `--json`），
  支持任意「d 位 / 每位 `[0, base)`」等概率数字玩法，默认排列3 输出逐位不变。
- `audit_external.py`：把自己记录的 CSV / JSONL 投注流水用**同一套**审计函数核对，
  全程本地，可选脱敏导出，绝不写回仓库发布数据。
- `pre_commit_checks.py` + `.pre-commit-config.yaml`：6 项静态自检，CI 与本地钩子共用。
- `combos.py` 纯计算层 + `main_lite.py` 轻量 GUI 入口 + `build_exe.py`：
  可构建不含 TensorFlow 的 Windows exe，详见 `BUILD.md`。
- `README.en.md` 完整英文文档、`CHANGELOG.md`、`records_template.csv`。

## 验证

| 检查 | 结果 |
|---|---|
| `python test_core.py` | 统计核心测试全绿（零 TF / 零 tkinter） |
| `python reproduce.py` | README 每个数字与数据来源逐项重算通过 |
| `python pre_commit_checks.py` | 6 项静态自检通过 |
| `python test_pipeline.py` | 本地全链路测试全绿（需 TF 2.13 与显示环境） |
| GitHub Actions | Python 3.11 / 3.13 双任务通过 |

干净 clone 后复现，只需：

```bash
pip install -r requirements-core.txt
python reproduce.py
```

## 边界与免责

`score` 是模型打分或排序依据，**不是中奖概率**；真实概率恒为玩法覆盖率。
本工具不提供任何投注建议，彩票的期望收益为负。数据来源与已知限制见
`DATA_SOURCES.md`。MIT 许可。

完整变更历史见 `CHANGELOG.md`。
