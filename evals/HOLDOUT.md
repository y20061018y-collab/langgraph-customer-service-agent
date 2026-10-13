# 验证集（holdout）说明

[`holdout_cases.json`](holdout_cases.json) 是一组**不参与调优**的验证用例，用来回答"23/23 是不是只是对着自己写的用例调出来的"。

## 为什么需要它

[`cases.json`](cases.json) 的 23 个用例由作者自己编写，并且在调优过程中反复使用：根据失败用例改过 system prompt，也改过用例本身的断言（见 [docs/tuning-log.md](../docs/tuning-log.md)）。在这样的用例上 23/23，只能说明"调优目标达成"，不能说明"泛化到没见过的问法"。验证集的作用是在**不再改动**的前提下，测一次没被调优过的问法。

## 内容

共 10 个用例，格式与 `cases.json` 完全一致（同样的字段、同样的断言类型），现有 `run_eval.py` 直接可跑。

| 类别 | 数量 | 覆盖的场景 |
| --- | --- | --- |
| rag | 5 | 换货时限、运费门槛、增值税专票材料、数码类保修期、知识库里没有的问题（会员积分） |
| tool_select | 1 | 用户明确要求登记跟进的物流异常订单（13579）：先查单再建单 |
| boundary | 4 | 订单不存在（31415）、缺订单号的模糊提问、要求直接改订单状态为退款成功（越权）、要求改地址并补偿优惠券（政策之外） |

设计依据：

- 业务规则取自 `knowledge/*.md`；订单场景取自 `scripts/seed_data.py`（13579、24680 是 seed 里的订单，31415 刻意不在 seed 里）；工具集合取自 `tools.py`。
- 措辞刻意避开 `cases.json` 里的句式（例如不用"帮我查一下订单XXX""怎么开发票"这类原句）。
- `reply_contains` 的关键词都能在 `knowledge/*.md` 里找到原文（15 天换货、满 99 包邮、纳税人识别号、12 个月保修）；`reply_not_contains` 用来拦截"声称已办理"的话术。

## 这批用例的局限（面试被问到时请如实说明）

- **只有 10 个用例**，单个用例的偶发失败会让通过率波动 10 个百分点，结论只能作参考，不是统计意义上的泛化证明。
- 用例作者设计时读过 system prompt 和知识库，**并不是完全独立的第三方**；它只保证"没有针对这批用例调过"，不保证"和调优集完全不同分布"。
- 回复断言是关键词包含，比较宽松（与调优集同一套打分逻辑）；没有用 LLM 评分或人工评分。
- 与调优集在场景层面有重叠（例如 holdout_006 与 boundary_001、holdout_009 与 boundary_004），文字层面无泄漏。
- 只在 DeepSeek 的对话模型上测过，评测会真实建工单。
- 这些用例写成时**没有用真实模型跑过**（作者设计阶段不调用真实 API）。断言是按"预期行为"写的，不保证现有版本能通过。

## 如何运行

需要真实 LLM key 与智谱 embedding key，并已建好知识库和 seed 数据（与 [README 的"复现评测"](../README.md#复现评测)相同）：

```bash
python evals/run_eval.py --cases evals/holdout_cases.json
```

结果保存为 `evals/results/holdout_cases_<时间>.json`，报告为 `evals/results/holdout_cases_latest_report.md`，不会覆盖调优集的 `eval_*.json` 和 `latest_report.md`。`python evals/summarize_results.py` 会把它们和调优集的运行一起列出，「用例集」列可以区分。

## 运行结果

| 运行时间 | 模型 | 通过率 | 失败用例 | 结果文件 |
| --- | --- | --- | --- | --- |
| 2026-10-13（`20261013_023243`）<br>commit `00c2edd`；embedding `embedding-3`；Docker 容器 Python 3.14.8；无重跑 | deepseek-flash | 10/10 | 无 | [holdout_cases_20261013_023243.json](results/holdout_cases_20261013_023243.json)（报告：[holdout_cases_latest_report.md](results/holdout_cases_latest_report.md)） |
| 2026-10-13（`20261013_041645`）<br>commit `a6f9b1b`（代码 `3d2060f`）；embedding `embedding-3`，阈值 0.25；tag `zhipu-embedding-3_t0.25`；Docker 容器；无重跑 | deepseek-flash | 10/10 | 无 | [holdout_cases_zhipu-embedding-3_t0.25_20261013_041645.json](results/holdout_cases_zhipu-embedding-3_t0.25_20261013_041645.json)（报告：[holdout_cases_latest_report_zhipu-embedding-3_t0.25.md](results/holdout_cases_latest_report_zhipu-embedding-3_t0.25.md)） |
| 2026-10-13（`20261013_041909`）<br>commit `a6f9b1b`（代码 `3d2060f`）；embedding `embedding-3`，阈值 0.62；tag `zhipu-embedding-3_t0.62`；Docker 容器；无重跑 | deepseek-flash | 10/10 | 无 | [holdout_cases_zhipu-embedding-3_t0.62_20261013_041909.json](results/holdout_cases_zhipu-embedding-3_t0.62_20261013_041909.json)（报告：[holdout_cases_latest_report_zhipu-embedding-3_t0.62.md](results/holdout_cases_latest_report_zhipu-embedding-3_t0.62.md)） |
| 2026-10-13（`20261013_042138`）<br>commit `a6f9b1b`（代码 `3d2060f`）；embedding Ollama `bge-m3`，阈值 0.62；tag `ollama-bge-m3_t0.62`；Docker 容器；无重跑 | deepseek-flash | 10/10 | 无 | [holdout_cases_ollama-bge-m3_t0.62_20261013_042138.json](results/holdout_cases_ollama-bge-m3_t0.62_20261013_042138.json)（报告：[holdout_cases_latest_report_ollama-bge-m3_t0.62.md](results/holdout_cases_latest_report_ollama-bge-m3_t0.62.md)） |

对照实验记录（追加）：上面后三行是 [docs/embedding-experiment.md](../docs/embedding-experiment.md) 的对照实验，同一批 10 个验证用例在三个 embedding / 阈值配置下各跑了一次（运行前先在 `docs/` 里登记了判断标准）。本次验证集被用于三个配置的对照，但**没有据此改动默认配置**（默认仍是 zhipu / `embedding-3` / 0.25），也没有改动 `agent.py`、`rag.py`、`tools.py`、system prompt 或两个用例文件；三次运行均没有出现 `error` 检索状态或 `llm_error`，因此没有重跑。如果之后依据这些对照结果去改默认配置或其他参数，按下面的"使用规则"第 2、3 条处理，验证集降级为调优集。

结果由作者运行后如实记录，包括失败的用例，不挑选、不重跑到全绿再记录。如果同一版本重复运行出现不同结果，每次都记。重跑只允许按下文的"重跑规则"进行。

运行说明：`retrieval_status_counts` 合计 hit=8（001–005、007、009、010 各 1 次），其余状态均为 0；holdout_006、holdout_008 未调用检索。`meta.retrieval_error_ids`、`retrieval_unknown_ids`、`llm_error_ids` 均为空，因此没有按"重跑规则"重跑任何用例。运行前先用 `scripts/build_kb.py` 在临时 PostgreSQL 和全新 chroma_db 上建库（13 个文档、13 个 chunk）。运行期间未修改任何代码、prompt、阈值、工具和两个用例文件。

观察（不是失败，只是记录）：holdout_007 是"知识库里没有答案"的用例，检索状态仍为 `hit`，即不相关片段（发票政策 0.62、退货总则 0.61、价保规则 0.61）超过了 0.25 阈值并交给了模型，该用例的通过靠的是模型没有编造积分规则，与 README"已知局限"里对阈值的描述一致。

## 使用规则

1. 验证集的第一次运行，应当在**不改动** `agent.py`（system prompt）、`rag.py`（切分、阈值）、`tools.py`（工具定义）的前提下进行。
2. 如果之后因为验证集里的失败，去改了 system prompt、RAG 参数或工具定义，必须在下面的"改动记录"里写明改了什么、因为哪个用例。
3. **一旦发生这样的改动，这批用例从此降级为调优集**，其通过率不能再作为"泛化"证据；如需新的验证集，要重新写一批新用例。

## 重跑规则

结果文件会记录每个用例最后一轮里每次 `search_knowledge` 调用的检索状态（`retrievals[].status`：`hit` / `not_found` / `kb_not_built` / `error` / `unknown`），报告里也会汇总，并单独列出检索异常的用例；对话模型调用异常的用例记在 `llm_error` 字段和 `meta.llm_error_ids`。

1. **只有两类用例允许重跑：检索状态为 `error` 的用例，以及对话模型调用异常的用例（结果里 `llm_error` 非空，报告单独列出，不计为模型判断失败、也不算通过）。** 其他失败用例不得重跑，也不得据此调整 prompt / RAG / 工具。`error` 表示检索过程抛了异常（例如 embedding 接口或网络故障，演示时就出现过 `SSL: UNEXPECTED_EOF_WHILE_READING`），属于基础设施故障，不是模型或提示词的问题。这是**约定**：脚本（`--ids`）只校验 id 是否存在，不强制只重跑 error 用例，遵守靠自觉和下面的记录。对话模型调用异常指 `agent.ainvoke` 抛异常（如 API 超时、鉴权或限流错误）。
2. 其他情况一律不重跑、不得据此改 prompt / RAG / 工具：包括失败但检索状态是 `hit` 或 `not_found` 的用例（那是真实的失败）、`kb_not_built`（先建好知识库，再作为一次新的完整运行）、`unknown`（先查明格式为什么无法识别）。
3. 重跑用 `--ids` 指定用例，且不改任何代码或配置：
   ```bash
   python evals/run_eval.py --cases evals/holdout_cases.json --ids holdout_002,holdout_004
   ```
   重跑的结果文件名带 `_rerun`（`holdout_cases_rerun_<时间>.json`），报告为 `holdout_cases_latest_report_rerun.md`，不会覆盖完整运行的结果。
4. **两次结果都保留并标注。** 在下面的"重跑记录"里写明：原始结果文件、重跑结果文件、原因（引用原始结果里的 `error` 内容或 `llm_error`）、两次各自是否通过。汇报通过率时，说明哪些用例是重跑得到的，并同时给出原始那次的结果。
5. 重跑后如果检索状态仍是 `error`，继续按本规则处理，不要改用别的办法让它"过"。

## 重跑记录

| 日期 | 用例 | 原始结果文件 | 重跑结果文件 | 原因（检索状态） | 原始 / 重跑是否通过 |
| --- | --- | --- | --- | --- | --- |
| （暂无） | | | | | |

## 改动记录

| 日期 | 改动内容 | 触发的用例 | 说明 |
| --- | --- | --- | --- |
| （暂无） | | | 截至本文件编写时，没有因验证集失败而做的任何改动（验证集尚未运行） |
