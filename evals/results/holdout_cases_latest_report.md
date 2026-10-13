# Agent 评测报告

- 运行时间：2026-10-13 02:32:43（耗时 40s）
- 用例集：holdout_cases.json
- 模型：deepseek-flash（temperature=0）
- 总用例：10，通过：10，**通过率 100%**
- 本次检索状态（search_knowledge 调用次数）：hit 8，not_found 0，kb_not_built 0，error 0，unknown 0

| 类别 | 用例数 | 通过 | 通过率 |
| --- | --- | --- | --- |
| RAG 检索 | 5 | 5 | 100% |
| 工具选择 | 1 | 1 | 100% |
| 边界拒绝 | 4 | 4 | 100% |