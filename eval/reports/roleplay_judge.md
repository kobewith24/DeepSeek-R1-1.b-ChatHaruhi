# 角色扮演 LLM-as-a-Judge 评测

- 裁判模型：`deepseek-flash`（temperature=0，A/B 盲评 + 交换顺序）
- 被测：`deepseek-r1:1.5b`（基座） vs `deepseek-r1-roleplay`（微调）
- 样本：30 条（6 角色 × 5 类问题）

## 胜负统计

| 结果 | 数量 |
|---|---|
| 微调胜 | 25 |
| 基座胜 | 1 |
| 平局 | 0 |
| 两次顺序矛盾（不稳定） | 4 |

## 各维度平均分（1~5）

| 模型 | 角色一致性 | 沉浸度 | 回答质量 |
|---|---|---|---|
| 基座 | 1.08 | 1.28 | 1.28 |
| 微调 | 2.12 | 3.22 | 2.22 |

## 按问题类型

| 类型 | 微调胜 | 基座胜 | 平 | 矛盾 |
|---|---|---|---|---|
| chat | 5 | 0 | 0 | 1 |
| knowledge | 5 | 0 | 0 | 1 |
| emotion | 6 | 0 | 0 | 0 |
| ooc | 4 | 0 | 0 | 2 |
| values | 5 | 1 | 0 | 0 |

## 逐条结果

| id | 角色 | 类型 | 胜者 |
|---|---|---|---|
| wxb-1 | 韦小宝 | chat | ft |
| wxb-2 | 韦小宝 | knowledge | ft |
| wxb-3 | 韦小宝 | emotion | ft |
| wxb-4 | 韦小宝 | ooc | inconsistent |
| wxb-5 | 韦小宝 | values | ft |
| hp-1 | Harry Potter | chat | ft |
| hp-2 | Harry Potter | knowledge | ft |
| hp-3 | Harry Potter | emotion | ft |
| hp-4 | Harry Potter | ooc | inconsistent |
| hp-5 | Harry Potter | values | base |
| ayaka-1 | 神里绫华 | chat | ft |
| ayaka-2 | 神里绫华 | knowledge | inconsistent |
| ayaka-3 | 神里绫华 | emotion | ft |
| ayaka-4 | 神里绫华 | ooc | ft |
| ayaka-5 | 神里绫华 | values | ft |
| txy-1 | 佟湘玉 | chat | ft |
| txy-2 | 佟湘玉 | knowledge | ft |
| txy-3 | 佟湘玉 | emotion | ft |
| txy-4 | 佟湘玉 | ooc | ft |
| txy-5 | 佟湘玉 | values | ft |
| qf-1 | 乔峰 | chat | inconsistent |
| qf-2 | 乔峰 | knowledge | ft |
| qf-3 | 乔峰 | emotion | ft |
| qf-4 | 乔峰 | ooc | ft |
| qf-5 | 乔峰 | values | ft |
| lyl-1 | 李云龙 | chat | ft |
| lyl-2 | 李云龙 | knowledge | ft |
| lyl-3 | 李云龙 | emotion | ft |
| lyl-4 | 李云龙 | ooc | ft |
| lyl-5 | 李云龙 | values | ft |