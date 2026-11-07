# 变更日志

本文件按时间倒序记录对外可见的变化。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循语义化版本；所有版本都保持同一个结论：**这个工具不预测号码**。

## [1.0.0] — 2026-11-07

首个可标记的开源稳定版。新增的都是「可验证性与可分发性」，没有新增任何预测能力。

### 新增

- **随机性检验的命令行接入**：`randomness.py` 支持 `--digits` / `--base` / `--data` /
  `--profile` / `--list-profiles` / `--json`；数据可来自 `.xlsx` / `.csv` / `.npy`。
  `PROFILES` 提供五种等概率数字型玩法规格模板，默认 `(3,10)` 输出逐位不变。
- **外部投注流水审计**：`audit_external.py` 把你自己在别处记录的 CSV / JSONL / JSON
  逐期流水，用与项目自身审计**完全相同**的函数（`audit.summarize` / `format_audit`）
  算出命中率、Wilson 区间、单边二项 p 值、返奖率与功效阈值。只读用户指定的文件，
  可选导出脱敏聚合；绝不写回仓库内的 `audit_aggregate.json` 或 `picks/`。
- **pre-commit 与项目自检**：`.pre-commit-config.yaml` + 零依赖的
  `pre_commit_checks.py`（6 项静态自检：私有文件、密钥字面量、`.gitignore` 规则、
  核心模块依赖边界、UTF-8 编码、文档资产）。CI 跑的是同一个脚本。
- **GUI 可打包分发**：抽出无 TF 依赖的 `combos.py` 纯计算层，新增 `main_lite.py`
  轻量入口（均匀随机选号，不加载神经网络）与 `build_exe.py` 构建脚本；
  `lite` / `full` 两种模式，产物不进仓库。详见 `BUILD.md`。
- **完整英文文档**：`README.en.md`，与中文版结构对齐。
- **发布资产**：本变更日志、`RELEASE_NOTES.md` 草稿、`records_template.csv` 模板；
  本地 tag `v1.0.0`。

### 变更

- `gui.py` 的选号结果标签改为可配置：完整版仍显示「模型打分」，轻量版显示
  「单注理论概率」，避免把均匀随机抽样误写成模型输出。
- `simulation.py` 不再从 `lottery_model` 顶层导入；`lottery_model.py` 重新导出
  `combos.py` 的实现，旧调用 `from lottery_model import iter_combos` 保持可用。
- `audit.format_audit()` 增加可选 `title`，默认措辞不变。
- CI 增加项目自检步骤与新资产的存在性检查。

## [0.3.0] — 2026-11-07 · `0396462`

- `randomness.analyze(data, digits, base)` 玩法通用化：重复形态用第二类 Stirling 数、
  和值用整数卷积精确推导，自由度随配置计算；默认 `(3,10)` 输出与旧版一致。
- 新增 `DATA_SOURCES.md` 与 `config.json.data_source`，`reproduce.py` 增加 7 项来源核对。
- `picks` 存档写入 `schema_version=2`；无版本字段的旧档按 v1 兼容，未来版本拒绝。
- README 嵌入两张结果图、补英文摘要，测试项数同步为 75 / 56 / 128。

## [0.2.0] — 2026-11-07 · `9a20c21`

- 引入 `.github/workflows/ci.yml`（Python 3.11 / 3.13 matrix）、`requirements-core.txt`
  依赖分层、`test_core.py` 零重依赖核心测试、README 状态徽章与持续集成章节。

## [0.1.1] — 2026-11-07 · `4cb502e`

- 九项随机性检验加入 Holm-Bonferroni 校正（控制 FWER），GUI 与报告同步口径。
- 新增 `audit_aggregate.json` 脱敏聚合数据与 `reproduce.py` 一键复现（49 项）。

## [0.1.0] — 2026-11-07 · `255a69c`

- 首次公开发布：排列3 分析与选号工具 v2.30，移除「预测下一期」输出、明确
  `score` 不是中奖概率、补 README 与 MIT LICENSE。
- 发布 7,770 期公开开奖数据、九项随机性检验、累积审计、Kronos 迁移否证实验。
