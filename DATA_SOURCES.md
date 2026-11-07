# 数据来源与口径（`历史数据.xlsx`）

本文件说明随仓库发布的 `历史数据.xlsx` 从哪里来、每一列是什么、如何更新、
以及有哪些已知限制。机器可核对的部分写在 `config.json` 的 `data_source` 段，
由 `reproduce.py` 逐项比对（期数、期号范围、列名、期号唯一性、开奖号格式）。

## 来源

| 项 | 内容 |
|---|---|
| 数据集 | 中国体育彩票「排列3」历史开奖号码 |
| 提供方 | 体彩官方公开接口 `webapi.sporttery.cn` |
| 接口 | `GET https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry` |
| 关键参数 | `gameNo=350133`（排列3）、`provinceId=0`、`pageSize`、`pageNo`、`isVerify=1` |
| 完整 URL 模板 | `config.json` → `fetch_url`，由 `data_processor.py` 填入分页参数 |
| 快照核验日期 | 2026-11-07 |
| 快照规模 | 7,770 期，`Period` 最小值 4001、最大值 26295 |

## 字段

| 列 | 类型 | 含义 |
|---|---|---|
| `Period` | 整数 | 期号。**文件内降序**（最新在最上）；读入后按升序（旧 → 新）组织 |
| `Numbers` | 文本 | 空格分隔的 3 个 `0–9` 数字，按开奖顺序排列，例如 `2 9 1` |

只有这两列。文件不含开奖时间、试机号、销售额、奖池或任何派奖信息；
`Numbers` 的顺序对「直选」有意义，对「组选」玩法由 `evaluation.canonical` 归一化。

**期号不连续**：排列3 期号跨年跳变（例如 `25099 → 26001`），所以不能假设
`base_period + 1` 一定存在。`settlement.load_draws_after` 只比较 `Period > base_period`。

## 采集与更新

- 增量更新由 GUI 的「更新数据」触发，走 `DataProcessor.fetch_and_update`。
- 按 `Period` 去重后追加新期，降序排列，先写临时 `.xlsx` 再原子替换，避免写坏原文件。
- **更新会整体重写工作簿**，因此不要手工往 xlsx 里加批注或额外工作表 —— 下次更新就会丢失。
  来源说明刻意放在本文件与 `config.json`，两者都是纯文本、可 diff、可随版本一起审查。
- 不做数据增强、不伪造开奖号、不填补缺失期号、不做插值；统计检验与训练直接使用原始开奖号。

## 快照与可复现性

仓库里的 xlsx 是**某一时点的静态快照**，不是实时数据。接口随时间增加新期，
因此更新数据后，以下三处必须一起重算，否则 `reproduce.py` 会直接失败：

1. `config.json` → `data_source.draw_count` / `period_min` / `period_max`；
2. `README.md`「先说结论」里的九项 p 值与期数；
3. `reproduce.py` → `EXPECTED.draw_count` / `EXPECTED.p_values`。

复现脚本同时复用这套数据做随机性检验，所以数据一变，统计结论本来就应该跟着变 ——
这正是它防止「文档数字悄悄漂移」的方式。

## 校验

```bash
pip install -r requirements-core.txt
python reproduce.py          # 含 7 项数据来源与口径核对
```

## 免责

这是公开接口的查询结果，**不是官方授权发布的数据集**，不保证完整性、准确性或实时性，
与彩票发行、销售机构无任何关联。仅用于统计分析与自我审计，
不构成投注建议，也不承诺任何收益提升。未满 18 周岁不得购买彩票。
