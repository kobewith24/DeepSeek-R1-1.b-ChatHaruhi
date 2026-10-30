# 角色扮演 LLM-as-a-Judge 评测

- 裁判模型：`deepseek-flash`（temperature=0，A/B 盲评 + 交换顺序后取平均）
- 被测：`deepseek-r1:1.5b`（微调前） vs `deepseek-r1-roleplay`（微调后）
- 样本：30 条（6 角色 × 5 类问题）

## 平均得分（1~5 分）

| 维度 | 微调前 | 微调后 | 提升 |
|---|---|---|---|
| 角色一致性 | 1.08 | 2.12 | +1.04 |
| 沉浸度（不出戏） | 1.28 | 3.22 | +1.94 |
| 回答质量 | 1.28 | 2.22 | +0.94 |
| **综合** | **1.22** | **2.52** | **+1.30** |

## 按问题类型（三项均分）

| 类型 | 微调前 | 微调后 | 提升 |
|---|---|---|---|
| chat | 1.14 | 2.75 | +1.61 |
| knowledge | 1.17 | 2.31 | +1.14 |
| emotion | 1.05 | 2.47 | +1.42 |
| ooc | 1.28 | 2.50 | +1.22 |
| values | 1.45 | 2.56 | +1.11 |

## 逐条得分

| id | 角色 | 类型 | 微调前 | 微调后 |
|---|---|---|---|---|
| ayaka-1 | 神里绫华 | chat | 1.00 | 3.67 |
| ayaka-2 | 神里绫华 | knowledge | 1.33 | 1.67 |
| ayaka-3 | 神里绫华 | emotion | 1.00 | 2.17 |
| ayaka-4 | 神里绫华 | ooc | 1.50 | 2.33 |
| ayaka-5 | 神里绫华 | values | 1.00 | 1.50 |
| hp-1 | Harry Potter | chat | 1.17 | 2.50 |
| hp-2 | Harry Potter | knowledge | 1.33 | 1.67 |
| hp-3 | Harry Potter | emotion | 1.00 | 1.50 |
| hp-4 | Harry Potter | ooc | 1.33 | 1.33 |
| hp-5 | Harry Potter | values | 3.50 | 2.33 |
| lyl-1 | 李云龙 | chat | 1.00 | 2.50 |
| lyl-2 | 李云龙 | knowledge | 1.17 | 3.17 |
| lyl-3 | 李云龙 | emotion | 1.33 | 3.83 |
| lyl-4 | 李云龙 | ooc | 1.33 | 3.50 |
| lyl-5 | 李云龙 | values | 1.00 | 3.00 |
| qf-1 | 乔峰 | chat | 1.00 | 2.67 |
| qf-2 | 乔峰 | knowledge | 1.00 | 2.50 |
| qf-3 | 乔峰 | emotion | 1.00 | 2.50 |
| qf-4 | 乔峰 | ooc | 1.00 | 3.67 |
| qf-5 | 乔峰 | values | 1.00 | 2.50 |
| txy-1 | 佟湘玉 | chat | 1.67 | 3.00 |
| txy-2 | 佟湘玉 | knowledge | 1.00 | 2.50 |
| txy-3 | 佟湘玉 | emotion | 1.00 | 2.83 |
| txy-4 | 佟湘玉 | ooc | 1.33 | 2.83 |
| txy-5 | 佟湘玉 | values | 1.00 | 3.17 |
| wxb-1 | 韦小宝 | chat | 1.00 | 2.17 |
| wxb-2 | 韦小宝 | knowledge | 1.17 | 2.33 |
| wxb-3 | 韦小宝 | emotion | 1.00 | 2.00 |
| wxb-4 | 韦小宝 | ooc | 1.17 | 1.33 |
| wxb-5 | 韦小宝 | values | 1.17 | 2.83 |