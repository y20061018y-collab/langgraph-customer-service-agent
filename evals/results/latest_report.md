# Agent 评测报告

- 运行时间：2026-10-12 15:51:49（耗时 76s）
- 用例集：cases.json
- 模型：deepseek-flash（temperature=0）
- 总用例：23，通过：23，**通过率 100%**
- 本次检索状态（search_knowledge 调用次数）：hit 7，not_found 0，kb_not_built 0，error 0，unknown 0

| 类别 | 用例数 | 通过 | 通过率 |
| --- | --- | --- | --- |
| 工具选择 | 6 | 6 | 100% |
| 边界拒绝 | 4 | 4 | 100% |
| 多轮记忆 | 4 | 4 | 100% |
| RAG 检索 | 4 | 4 | 100% |
| 鲁棒性 | 5 | 5 | 100% |