# Embedding 对照实验：判断标准（预先登记）

本文件在**任何实验数据产生之前**定稿并提交。之后只允许在「结果」章节填入数据；方法、判断标准、采用规则、局限这几节如需修改，必须在文末「修订记录」里写明时间和原因，不得悄悄改动。

## 1. 目的与背景

目的：在同一套代码、同一个知识库、同一批用例上，比较三种 embedding 配置的**检索阈值是否起作用**，以及换 embedding 后 Agent 的评测通过数**有没有变差**。不判断哪个 embedding 更好。

背景（均为已有记录，不是本实验的新数据）：

- **阈值 0.25 对 `embedding-3` 起不到过滤作用，有两处证据：**
  1. 演示时问"你们支持分期付款吗？"（`knowledge/` 中没有该内容），检索返回发票政策 0.57、价保规则 0.54、发货时效 0.53，三个无关片段都高于 0.25，没有触发"未找到"降级（见 README"已知局限"和 `docs/demo.gif`）。
  2. 验证集 holdout_007（会员积分，知识库里没有答案）的检索状态是 `hit`：发票政策 0.62、退货总则 0.61、价保规则 0.61 超过阈值并交给了模型（见 [evals/HOLDOUT.md](../evals/HOLDOUT.md)"运行结果"一节的观察）。
  两处都是单个问题的观察，不是分布统计；本实验用校准集补上分布。
- **SSL 偶发断连**（`SSL: UNEXPECTED_EOF_WHILE_READING`）：只是背景。代码里已加重试和日志，但原因仍未定位。**本实验不对它下任何结论**；运行中出现的 `error` 状态按 HOLDOUT.md 的重跑规则处理（见下）。

## 2. 方法

### 2.1 三个配置

| 编号 | embedding | 相似度阈值 |
| --- | --- | --- |
| (a) | 智谱 `embedding-3`（1024 维） | 0.25（现行默认值） |
| (b) | 智谱 `embedding-3`（1024 维） | `T_zhipu`，由校准集得出 |
| (c) | 本地 Ollama `bge-m3` | `T_bge`，由校准集得出 |

- 代码版本固定为 `3d2060f`（`embedding-bge-m3` 分支，代码冻结）。实验期间不改 `rag.py`、`agent.py`、`tools.py`、system prompt、两个用例文件。
- (a) 在新代码上**重跑**一次，作为同批次对照；不直接引用历史结果当对照。
- 对话模型沿用 `deepseek-flash`（temperature=0），知识库 13 篇文档不变。

### 2.2 阈值的得出

阈值只由校准集 [`evals/threshold_probe.json`](../evals/threshold_probe.json) 按 `evals/threshold_probe.py` 里**已经固定的规则**得出，不手调，**不得使用 holdout（验证集）**：

- 记 A = 可回答问题的 top-1 相似度，U = 无答案问题（业务相近 + 业务无关合并）的 top-1 相似度。
- 可分（max U < min A）：T = 两者中点，四舍五入到 0.01。
- 不可分：T = floor(min A × 100) / 100，即保住全部可回答问题。
- 规则之外的边界兜底（取整后不再严格分开、浮点进位等）以 `choose_threshold` 的实现为准，已有离线测试覆盖。

每个 embedding 各建一次库、各跑一次校准：`python scripts/build_kb.py --rebuild` 后运行 `python evals/threshold_probe.py`。校准集是作者编写的，不是验证集；它在运行前已提交（见 git 历史）。

校准集构成：可回答 16 条（自然问句 8、关键词短句 8，覆盖 `knowledge/` 全部 13 篇文档）；无答案 16 条，其中业务相近（near）8 条、业务无关（off）8 条，各自又是自然问句 4、关键词短句 4。关键词短句仿照 Agent 改写后送检的真实 query（来自调优集结果里的 `retrievals[].query`）。每条无答案问题都在 `kb_check` 字段里记录了对知识库原文的 grep 核对。

### 2.3 评测运行

每个配置在**调优集（`cases.json`，23 条）和验证集（`holdout_cases.json`，10 条）各跑一次**，用 `--tag` 区分，例如：

```bash
python evals/run_eval.py --tag zhipu-embedding-3_t0.25
python evals/run_eval.py --cases evals/holdout_cases.json --tag zhipu-embedding-3_t0.25
```

- 每次运行前，用对应配置 `--rebuild` 重建向量库，并确认 `threshold_probe` / `run_eval` 记录的 embedding 元数据与配置一致。
- 只有两类用例允许重跑：检索状态为 `error` 的用例，以及 `llm_error` 非空的用例（HOLDOUT.md"重跑规则"）。用 `--ids` 重跑，**原始结果和重跑结果两份都保留**，汇报时同时给出；其他失败（检索是 `hit` / `not_found` 的真实失败）一律不重跑、不据此改任何东西。
- 每个配置每个集合只做一次完整运行。

## 3. 判断标准（预先登记）

### 3.1 校准后能过滤无关内容

在校准集上，对 (b) 和 (c) **各自**用自己的 T 计算（以 top-1 相似度 < T 判为被过滤；(a) 的 0.25 作为参照一并报告，不设通过条件）：

- **通过条件：业务相近（near）无答案问题中，top-1 被阈值过滤的数量 ≥ 7/8。** 达不到则该配置"校准后仍不能过滤业务相近的无关问题"。
- 只报告、不作通过条件：
  - 业务无关（off）无答案问题被过滤的数量；
  - 自然问句、关键词短句两类各自的被过滤数量（无答案）与保留数量（可回答）；
  - 可回答问题的保留率（T 是在这个集合内拟合出来的，是 in-sample，只描述拟合结果，不是独立验证）；
  - 校准集是否严格可分（max U < min A），单独报告；
  - 可回答问题 top-1 来源是否正确（`answerable_top1_correct`）。

### 3.2 评测不退步

- 每个配置在调优集和验证集的**通过数都 ≥ 同批次 (a) 的通过数**。
- 同时列出与历史基线（调优集 23/23、验证集 10/10）的对比。
- 某配置通过数少于 (a) 时，逐条对照该用例的 `retrievals` 与 `tool_outputs`，把失败分成两类并**分开写，不做合并结论**：
  1. 检索变了（命中的片段 / 状态 / 相似度与 (a) 不同，导致模型看到的材料不同）；
  2. 模型输出波动（检索结果与 (a) 基本一致，但回复不同）。

### 3.3 次要观察（只陈述，不打分）

- 无答案用例（如 holdout_007）的检索状态是否变成 `not_found`；
- 可回答的 rag 用例是否仍为 `hit`，次要片段被阈值过滤掉几个。

### 3.4 不下的结论

**不对"哪个 embedding 更好"下结论。** 本实验只回答：阈值校准后能否过滤业务相近的无关问题（3.1），以及换配置后评测通过数是否没变差（3.2）。

## 4. 采用规则

- 本分支**不改默认值**：默认仍是 zhipu / `embedding-3` / 0.25。
- 若之后要改默认配置，必须作为单独的决定提出，写明理由。
- holdout 在本实验里已被用于对比配置，如果据此选型，按 [HOLDOUT.md](../evals/HOLDOUT.md) 的使用规则，**验证集降级为调优集**，其通过率不再作为泛化证据，需要新写一批验证用例。

## 5. 局限

- 23/23、10/10 已经满分：这两套 eval 只能检验"没变差"，**分辨不出"变好"**。
- 校准集由作者编写，每个子类只有 8 条，粒度很粗；"near 类 ≥ 7/8"意味着允许 1 条漏网，换一批问题结果可能不同。
- 阈值 T 是在校准集内拟合出来的（in-sample），可回答问题的保留率必然接近 100%，不能当作独立验证。
- Agent 实际送检的是模型改写后的 query，校准集里的关键词短句只是仿照，分布不会完全相同。
- 阈值是对返回的每个片段分别过滤，而校准只看 top-1 的相似度，不能完全代表次要片段的过滤情况。
- 每个配置、每个集合只跑一次，LLM 输出有波动；验证集只有 10 条，单条用例就是 10 个百分点。
- 评测会真实建工单，只在一个对话模型（DeepSeek）上测过。
- 校准集的 near 类问题是作者按"知识库里没有"逐条 grep 选出的，不代表真实用户会问的无答案问题的分布。
- 探测结果文件的 meta 里没有记录 base_url 主机和相似度阈值（探测脚本不经过阈值）；3b-1 是用启动日志 `rag.log_index_status()` 的输出核对的主机和阈值，评测结果 meta 里则有 `similarity_threshold`。
- (b) 与 (c) 的 T 都是 0.62，原因是两个配置的可回答问题 top-1 最小值接近（embedding-3 为 0.6215，bge-m3 为 0.6209），两者按规则向下取整到 0.01 后都得到 0.62；这不表示两个配置的相似度分布相同。

## 6. 结果

> 以下数据由脚本从结果 JSON 生成，只记录数据，不在本节下结论。

### 6.1 校准分布与选定阈值（`threshold_probe`，运行于 2026-10-13，代码 `3d2060f`，校准集 `a6f9b1b`）

| 项目 | (b) embedding-3 | (c) bge-m3 |
| --- | --- | --- |
| 结果文件 | `evals/results/threshold_probe_zhipu-embedding-3_20261013_041134.json` | `evals/results/threshold_probe_ollama-bge-m3_20261013_041154.json` |
| 结果 meta：provider / model / 库内维度 / 查询维度 / chunk 数 | zhipu / embedding-3 / 1024 / 1024 / 13 | ollama / bge-m3 / 1024 / 1024 / 13 |
| 可回答 A：top-1 min / median / max | 0.6215 / 0.7331 / 0.8087 | 0.6209 / 0.7067 / 0.8057 |
| 无答案 U：top-1 min / median / max | 0.3143 / 0.4821 / 0.6729 | 0.2254 / 0.4735 / 0.6301 |
| 无答案 near：min / median / max | 0.49 / 0.5976 / 0.6729 | 0.5118 / 0.5955 / 0.6301 |
| 无答案 off：min / median / max | 0.3143 / 0.4552 / 0.4742 | 0.2254 / 0.3427 / 0.4351 |
| 可回答·自然问句 / 关键词短句：min / median / max | 0.6215 / 0.6647 / 0.7567 ；0.727 / 0.7639 / 0.8087 | 0.6209 / 0.7008 / 0.7711 ；0.6639 / 0.7194 / 0.8057 |
| 无答案·自然问句 / 关键词短句：min / median / max | 0.3143 / 0.4821 / 0.59 ；0.4516 / 0.5326 / 0.6729 | 0.2254 / 0.4553 / 0.6047 ；0.2753 / 0.4772 / 0.6301 |
| 是否严格可分（max U < min A） | False | False |
| T 及所用规则分支 | 0.62（不可分：min A 向下取整到 0.01（保住全部可回答问题）） | 0.62（不可分：min A 向下取整到 0.01（保住全部可回答问题）） |
| near 被过滤（标准 ≥ 7/8） | 6/8 | 6/8 |
| off 被过滤 | 8/8 | 8/8 |
| 无答案合计被过滤 | 14/16 | 14/16 |
| 自然问句：无答案被过滤 / 可回答保留 | 8/8 / 8/8 | 8/8 / 8/8 |
| 关键词短句：无答案被过滤 / 可回答保留 | 6/8 / 8/8 | 6/8 / 8/8 |
| 可回答合计保留（in-sample） | 16/16 | 16/16 |
| 可回答 top-1 来源正确 | 16/16 | 12/16 |

(a) 在 0.25 下的同口径数字（用 (b) 的相似度分布，因为 (a) 与 (b) 是同一个 embedding）：near 被过滤 0/8，off 被过滤 0/8；无答案问题 top-1 最小值为 0.3143，高于 0.25。

未被阈值过滤的 near 问题（top-1 ≥ T）：(b) cal_n05（0.6729）、cal_n07（0.6202）；(c) cal_n05（0.6301）、cal_n08（0.6267）。

两个配置的 near 过滤数都没有达到预先登记的 ≥ 7/8（3.1）。按登记的规则，如实记录为"校准后仍不能过滤业务相近的无关问题"，不据此调整规则或校准集。

#### 逐条 top-1（从结果 JSON 读出；预期来源列只对可回答问题有意义；✓/✗ 为 top-1 来源是否等于预期来源）

| id | style | group | (b) embedding-3 top-1 | (b) 来源 | (c) bge-m3 top-1 | (c) 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| cal_a01 | natural | 可回答 | 0.6636 | 01-退货政策总则 ✓ | 0.6209 | 03-退款流程与时效 ✗（预期 01-退货政策总则，预期来源排第 2） |
| cal_a02 | natural | 可回答 | 0.7355 | 02-七天无理由退货 ✓ | 0.7113 | 02-七天无理由退货 ✓ |
| cal_a03 | natural | 可回答 | 0.7567 | 03-退款流程与时效 ✓ | 0.7711 | 03-退款流程与时效 ✓ |
| cal_a04 | natural | 可回答 | 0.6659 | 04-退货运费规则 ✓ | 0.635 | 12-签收与验货 ✗（预期 04-退货运费规则，预期来源排第 2） |
| cal_a05 | natural | 可回答 | 0.6349 | 05-发货时效 ✓ | 0.7402 | 05-发货时效 ✓ |
| cal_a06 | natural | 可回答 | 0.6561 | 07-物流异常处理 ✓ | 0.7022 | 12-签收与验货 ✗（预期 07-物流异常处理，预期来源排第 2） |
| cal_a07 | natural | 可回答 | 0.7033 | 10-质量问题处理 ✓ | 0.6809 | 10-质量问题处理 ✓ |
| cal_a08 | natural | 可回答 | 0.6215 | 13-价保规则 ✓ | 0.6995 | 13-价保规则 ✓ |
| cal_a09 | keyword | 可回答 | 0.7307 | 06-运费规则 ✓ | 0.717 | 06-运费规则 ✓ |
| cal_a10 | keyword | 可回答 | 0.7838 | 08-换货流程 ✓ | 0.6639 | 04-退货运费规则 ✗（预期 08-换货流程，预期来源排第 2） |
| cal_a11 | keyword | 可回答 | 0.7708 | 09-保修条款 ✓ | 0.6914 | 09-保修条款 ✓ |
| cal_a12 | keyword | 可回答 | 0.7398 | 11-发票政策 ✓ | 0.7218 | 11-发票政策 ✓ |
| cal_a13 | keyword | 可回答 | 0.727 | 12-签收与验货 ✓ | 0.7308 | 12-签收与验货 ✓ |
| cal_a14 | keyword | 可回答 | 0.757 | 03-退款流程与时效 ✓ | 0.7825 | 03-退款流程与时效 ✓ |
| cal_a15 | keyword | 可回答 | 0.8087 | 02-七天无理由退货 ✓ | 0.701 | 02-七天无理由退货 ✓ |
| cal_a16 | keyword | 可回答 | 0.806 | 13-价保规则 ✓ | 0.8057 | 13-价保规则 ✓ |
| cal_n01 | natural | near | 0.49 | 13-价保规则 | 0.5118 | 11-发票政策 |
| cal_n02 | natural | near | 0.5686 | 13-价保规则 | 0.5674 | 13-价保规则 |
| cal_n03 | natural | near | 0.59 | 07-物流异常处理 | 0.5864 | 12-签收与验货 |
| cal_n04 | natural | near | 0.5872 | 06-运费规则 | 0.6047 | 06-运费规则 |
| cal_n05 | keyword | near | 0.6729 | 13-价保规则 | 0.6301 | 01-退货政策总则 |
| cal_n06 | keyword | near | 0.6078 | 09-保修条款 | 0.5192 | 08-换货流程 |
| cal_n07 | keyword | near | 0.6202 | 05-发货时效 | 0.6102 | 09-保修条款 |
| cal_n08 | keyword | near | 0.6052 | 06-运费规则 | 0.6267 | 03-退款流程与时效 |
| cal_o01 | natural | off | 0.4742 | 05-发货时效 | 0.3987 | 12-签收与验货 |
| cal_o02 | natural | off | 0.4023 | 07-物流异常处理 | 0.363 | 11-发票政策 |
| cal_o03 | natural | off | 0.3143 | 09-保修条款 | 0.2254 | 08-换货流程 |
| cal_o04 | natural | off | 0.3634 | 06-运费规则 | 0.2832 | 09-保修条款 |
| cal_o05 | keyword | off | 0.4588 | 06-运费规则 | 0.3225 | 11-发票政策 |
| cal_o06 | keyword | off | 0.4587 | 03-退款流程与时效 | 0.4351 | 13-价保规则 |
| cal_o07 | keyword | off | 0.4516 | 06-运费规则 | 0.2753 | 11-发票政策 |
| cal_o08 | keyword | off | 0.4599 | 05-发货时效 | 0.3628 | 05-发货时效 |

### 6.2 评测通过数（运行于 2026-10-13，代码 `3d2060f`，聊天模型 `deepseek-flash`；6 次完整运行，无重跑）

| 配置 | 数据集 | 通过 | 总数 | 各类别通过数 | 检索状态统计（全部用例合计） | llm_error_ids | retrieval_error_ids | 结果文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) embedding-3 @0.25 | 调优集 | 23 | 23 | tool_select 6/6 boundary 4/4 memory 4/4 rag 4/4 robustness 5/5 | hit=7 not_found=0 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `eval_zhipu-embedding-3_t0.25_20261013_041519.json`（报告 `latest_report_zhipu-embedding-3_t0.25.md`） |
| (a) embedding-3 @0.25 | 验证集 | 10 | 10 | tool_select 1/1 boundary 4/4 rag 5/5 | hit=7 not_found=0 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `holdout_cases_zhipu-embedding-3_t0.25_20261013_041645.json`（报告 `holdout_cases_latest_report_zhipu-embedding-3_t0.25.md`） |
| (b) embedding-3 @0.62 | 调优集 | 23 | 23 | tool_select 6/6 boundary 4/4 memory 4/4 rag 4/4 robustness 5/5 | hit=7 not_found=0 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `eval_zhipu-embedding-3_t0.62_20261013_041745.json`（报告 `latest_report_zhipu-embedding-3_t0.62.md`） |
| (b) embedding-3 @0.62 | 验证集 | 10 | 10 | tool_select 1/1 boundary 4/4 rag 5/5 | hit=8 not_found=1 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `holdout_cases_zhipu-embedding-3_t0.62_20261013_041909.json`（报告 `holdout_cases_latest_report_zhipu-embedding-3_t0.62.md`） |
| (c) bge-m3 @0.62 | 调优集 | 23 | 23 | tool_select 6/6 boundary 4/4 memory 4/4 rag 4/4 robustness 5/5 | hit=7 not_found=0 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `eval_ollama-bge-m3_t0.62_20261013_042016.json`（报告 `latest_report_ollama-bge-m3_t0.62.md`） |
| (c) bge-m3 @0.62 | 验证集 | 10 | 10 | tool_select 1/1 boundary 4/4 rag 5/5 | hit=7 not_found=3 kb_not_built=0 error=0 unknown=0 | 无 | 无 | `holdout_cases_ollama-bge-m3_t0.62_20261013_042138.json`（报告 `holdout_cases_latest_report_ollama-bge-m3_t0.62.md`） |

同批次对比（通过数，括号内为相对同批次 (a) 的差值）与历史基线：

| 配置 | 调优集（基线 23/23） | 验证集（基线 10/10） |
| --- | --- | --- |
| (a) embedding-3 @0.25 | 23/23（相对 (a)：+0；相对历史基线：+0） | 10/10（相对 (a)：+0；相对历史基线：+0） |
| (b) embedding-3 @0.62 | 23/23（相对 (a)：+0；相对历史基线：+0） | 10/10（相对 (a)：+0；相对历史基线：+0） |
| (c) bge-m3 @0.62 | 23/23（相对 (a)：+0；相对历史基线：+0） | 10/10（相对 (a)：+0；相对历史基线：+0） |

没有任何用例触发重跑（`retrieval_error_ids`、`retrieval_unknown_ids`、`llm_error_ids` 在 6 次运行中均为空），因此没有重跑文件。

#### 逐用例对照（✓/✗ = 通过/失败；其后是该用例的检索状态，括号内为结果里记录的、阈值过滤后保留的命中片段数；多次检索用 + 连接；未调用检索写“无检索”）

**调优集**

| 用例 | 类别 | (a) @0.25 | (b) @0.62 | (c) bge-m3 @0.62 |
| --- | --- | --- | --- | --- |
| tool_select_001 | tool_select | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| tool_select_002 | tool_select | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| tool_select_003 | tool_select | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| tool_select_004 | tool_select | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| tool_select_005 | tool_select | ✓ hit(3) | ✓ hit(3) | ✓ hit(1) |
| tool_select_006 | tool_select | ✓ hit(3) | ✓ hit(1) | ✓ hit(1) |
| boundary_001 | boundary | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| boundary_002 | boundary | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| boundary_003 | boundary | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| boundary_004 | boundary | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| memory_001 | memory | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| memory_002 | memory | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| memory_003 | memory | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| memory_004 | memory | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| rag_001 | rag | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| rag_002 | rag | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| rag_003 | rag | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| rag_004 | rag | ✓ hit(3) | ✓ hit(1) | ✓ hit(1) |
| robust_001 | robustness | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| robust_002 | robustness | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| robust_003 | robustness | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| robust_004 | robustness | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| robust_005 | robustness | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |

**验证集**

| 用例 | 类别 | (a) @0.25 | (b) @0.62 | (c) bge-m3 @0.62 |
| --- | --- | --- | --- | --- |
| holdout_001 | rag | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| holdout_002 | rag | ✓ hit(3) | ✓ hit(3) | ✓ hit(2)+hit(3) |
| holdout_003 | tool_select | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| holdout_004 | rag | ✓ hit(3) | ✓ hit(1) | ✓ hit(1) |
| holdout_005 | rag | ✓ hit(3) | ✓ hit(1) | ✓ hit(1) |
| holdout_006 | boundary | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| holdout_007 | rag | ✓ hit(3) | ✓ not_found(0)+hit(2) | ✓ not_found(0)+not_found(0) |
| holdout_008 | boundary | ✓ 无检索 | ✓ 无检索 | ✓ 无检索 |
| holdout_009 | boundary | ✓ hit(3) | ✓ hit(3) | ✓ hit(3) |
| holdout_010 | boundary | ✓ hit(3) | ✓ hit(3)+hit(1) | ✓ hit(1)+not_found(0) |

### 6.3 通过数少于 (a) 的用例分析

(b)、(c) 在调优集和验证集的通过数都与同批次 (a) 相同（均为全部通过），没有“通过数少于 (a)”的用例，本节无需分类。

### 6.4 次要观察（只陈述）

知识库里没有答案的用例（调优集 rag_003 天气、验证集 holdout_007 会员积分）的检索：

| 用例 | (a) @0.25 | (b) @0.62 | (c) bge-m3 @0.62 |
| --- | --- | --- | --- |
| rag_003 | 无检索 | 无检索 | 无检索 |
| holdout_007 | query「会员积分 积分能否抵现金使用 积分规则」→ hit：11-发票政策 0.61、13-价保规则 0.61、01-退货政策总则 0.6 | query「会员积分 积分能否抵现金使用」→ not_found：无片段；query「会员积分规则 抵现 优惠政策」→ hit：13-价保规则 0.64、06-运费规则 0.63 | query「会员积分 积分抵现金 积分使用规则」→ not_found：无片段；query「会员积分制度 积分兑换 抵扣」→ not_found：无片段 |

可回答的 rag 用例（category=rag，不含上面两条）在三个配置下的检索状态与保留的片段数（过滤前的片段数、被阈值过滤掉的个数在结果里**未记录**，不推算）：

| 用例 | (a) @0.25 | (b) @0.62 | (c) bge-m3 @0.62 |
| --- | --- | --- | --- |
| rag_001 | hit(3) | hit(3) | hit(3) |
| rag_002 | hit(3) | hit(3) | hit(3) |
| rag_004 | hit(3) | hit(1) | hit(1) |
| holdout_001 | hit(3) | hit(3) | hit(3) |
| holdout_002 | hit(3) | hit(3) | hit(2)+hit(3) |
| holdout_004 | hit(3) | hit(1) | hit(1) |
| holdout_005 | hit(3) | hit(1) | hit(1) |


## 修订记录

| 日期 | 修改内容 | 原因 |
| --- | --- | --- |
| （暂无） | | 首次提交，先于任何实验数据 |
