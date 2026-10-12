# 企业客服 & 工单处理 Agent —— V0

这是旗舰项目的第一版：最小闭环，能跑通就是唯一目标，先不求全。

## 场景

用户："我的订单9527为什么还没发货，如果物流异常帮我创建工单"

Agent 需要自己完成：

```
理解任务 → 调 query_order 查订单 → 发现物流异常
    → 调 search_knowledge 确认处理规则 → 判断需要人工介入
    → 调 create_ticket 创建工单 → 返回处理结果
```

## 目录结构

```
agent-v0/
├── tools.py        # 3个mock工具：query_order / search_knowledge / create_ticket
├── llm.py          # 可插拔LLM工厂，换厂商只改.env，不改代码
├── agent.py        # 手写LangGraph StateGraph —— 核心就在这里
├── main.py         # FastAPI，暴露 POST /chat
├── test_agent.py   # 不需要真实API Key也能验证图结构对不对
├── requirements.txt
└── .env.example
```

## 为什么这么设计（这是你要能讲清楚的部分）

**为什么手写 StateGraph，不用 `create_react_agent`？**
`create_react_agent` 是 LangGraph 官方封装好的一行代码方案，能用但你说不清楚
"状态是怎么流转的"。手写之后，`agent.py` 里的图结构就是：

```
START → agent 节点 →(检查最后一条消息有没有 tool_calls?)
            ├─ 有 → tools 节点 → 回到 agent 节点（循环）
            └─ 没有 → END
```

这个循环本质上就是 ReAct（Reasoning + Acting）模式：模型每次决定"要不要调工具、
调哪个"，工具执行完把结果喂回去，模型接着判断，直到它觉得可以给最终答案为止。

```

**为什么三个工具都是假数据？**
V0 阶段要验证的是"Agent 的决策逻辑对不对"，不是"数据库连不连得上"。工具的
输入输出接口（函数签名）保持和未来真实版本一致，以后接真数据库时只需要改
`tools.py` 内部实现，`agent.py` 完全不用动——这就是面向接口设计的好处。

**为什么 LLM 要做成可插拔？**
`llm.py` 用的是 `ChatOpenAI`，但通过 `.env` 里的 `LLM_BASE_URL` 可以指向任何
OpenAI 兼容接口（智谱GLM、DeepSeek等）。这样你后面想换成用 Z.ai 的 GLM 来跑
这个 Agent，只改一行配置，不用重写任何业务代码。

## 怎么跑起来

```bash
pip install -r requirements.txt
cp .env.example .env   # 然后把 LLM_API_KEY 换成你自己的真实key
uvicorn main:app --reload
```

不想接真实API Key也能先验证图对不对：

```bash
python test_agent.py
```

这个测试用了一个"脚本化假LLM"，按预设顺序模拟"决定调用三个工具"的过程，
只验证 LangGraph 的状态流转对不对，不测试模型智不智能——这两件事要分开测。

## 接好真实API Key之后，测试用例

```bash
# 正常订单
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "我的订单12345到哪了"}'

# 物流异常，应该触发建工单
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "我的订单9527为什么还没发货，如果物流异常帮我创建工单"}'

# 纯知识库问题，不应该调用query_order
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "订单待发货多久算正常"}'
```

第三个用例是个好的"审计点"：如果 Agent 不管问什么都习惯性先查订单，
说明 system prompt 或者模型本身的判断力有问题，这个之后可以写进
PROJECT_AUDIT 里当作一个真实发现的问题。

## 这一版故意没做的事（V1 之后再加，别提前塞）

| 版本 | 加什么 |
| --- | --- |
| V1 | 已完成：Tool Calling 闭环 |
| V2 | 已完成：LangGraph 手写工作流 |
| V3 | 把 search_knowledge 换成真实 Embedding + 向量检索 |
| V4 | 已完成（V6 实现）：基于会话的跨请求多轮 Memory |
| V5 | 已完成：query_order / create_ticket 接真实 PostgreSQL |
| V6 | 已完成：多轮对话记忆（session + PostgreSQL 持久化 + 滑动窗口） |
| V7 | 已完成：自动化评测体系（23 用例 × 5 类，真实 LLM 批量断言） |
| V8 | HITL：高风险操作前先请求人工确认 |
| V8 | 权限 / 日志 / Trace |
| V9 | Redis / 并发处理 |
| V10 | Docker 部署 |

## 安全提醒

`.env` 不要传给任何AI、不要提交到git。现在 `.env.example` 里只有占位符，是安全的。

---

# V3：真实 RAG 知识库

V3 只做了一件事：把 `search_knowledge` 从"关键词假匹配"升级为真实的向量检索 RAG。
Agent 图结构、另外两个工具、`/chat` 接口全部零改动——V0 时把工具做成"面向接口"，
这一版兑现了它的收益。

## 架构

```
knowledge/*.md (13篇售后政策)
      │ ① 加载：rag.load_documents()
      ▼
原始文档 ──② 切分：rag.chunk_text()（结构感知 + 长度兜底）──► chunks
      │ ③ 向量化：rag.embed_texts() → POST {EMBEDDING_BASE_URL}/embeddings
      │            （智谱 embedding-3，OpenAI 兼容协议，1024 维）
      ▼
ChromaDB 持久化 ./chroma_db（cosine 相似度）
      │
      │ ④ 检索：rag.search(query, top_k=3)
      │    query 向量化 → 余弦相似度 → 相似度阈值过滤 → top_k
      ▼
带来源标注的上下文片段 ──► search_knowledge 工具返回给 Agent ──► 组织成回复
```

数据流分两条线：**建库线**（①②③，只跑一次，`python scripts/build_kb.py`）
和**查询线**（④，每次调用工具时执行）。

## 怎么跑

```bash
pip install -r requirements.txt
# .env 里配好 embedding（见下文"配置"）
python scripts/build_kb.py            # 幂等，已有数据会跳过
python scripts/build_kb.py --rebuild  # 知识文档改动后全量重建
uvicorn main:app --reload
```

验收测试：

```bash
# 问题在知识库里 → 回复内容来自知识库，且 /chat 的 tool_calls 轨迹里
# search_knowledge 的输出带（来源: xx，相似度 xx）标注
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "你们的退货政策是什么"}'

# 问题不在知识库里 → 相似度低于阈值被过滤，Agent 回复"不知道"而不是编造
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "太阳系有几大行星"}'
```

## 配置（.env）

```ini
EMBEDDING_MODEL_NAME=embedding-3        # 向量化模型
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
EMBEDDING_API_KEY=                      # 智谱开放平台的 key
EMBEDDING_DIMENSIONS=1024               # 可选：256/512/1024/2048
EMBEDDING_SIMILARITY_THRESHOLD=0.25     # 可选：低于此相似度判为未找到，0 关闭
```

`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` 留空时复用 `LLM_*` 的配置（都是
OpenAI 兼容协议）。**注意**：embedding-3 只在智谱开放平台提供；如果 LLM 配的
是 DeepSeek 等其他厂商（DeepSeek 没有 embedding 接口），必须单独填智谱的 key。

## 每个设计决定的原因（面试可讲）

**为什么不用 LangChain 的加载器 / TextSplitter / VectorStore？**
这三个封装各自只省 10 行代码，但代价是每个环节变成黑盒。自己写之后：
切分时能讲清"为什么 chunk_size 是 350、为什么要 overlap"；检索时能讲清
"cosine 和 l2 差在哪、为什么向量检索的 top_k 永远有结果所以必须加阈值"。
这版刻意保持朴素（段落合并 + 滑动窗口兜底），参数都是讲得出理由的经验值。

**为什么 embedding 用 httpx 直接调 /embeddings，而不是 SDK 或 LangChain Embeddings？**
OpenAI 兼容的 embedding 接口就是一个 POST `{model, input, dimensions}`，
一层薄 HTTP 封装，换厂商只改 base_url/key。SDK 会带来厂商耦合，LangChain
封装则又把这一层变黑盒。分批（16条/批）是为了绕开接口的单次条数限制。

**为什么选 ChromaDB？**
本地单机、原生支持持久化（`PersistentClient`）、API 是"存 embedding + 元数据，
查 top_k"三个操作，没有多余抽象。数据量大了之后再考虑迁移 pgvector/Milvus，
接口上只需要换 `get_collection()` 内部实现。

**为什么 cosine 而不是默认的 l2？**
余弦相似度只比较向量方向（语义），不比较模长（各维度数值幅度）。两个语义
相近但长短不同的文本，cosine 距离小而 l2 距离可能大。文本检索的标准选择。

**为什么建库要幂等（count>0 就跳过）？**
建库要调 embedding 接口，花钱且耗时；服务启动时如果误触发建库，不应该
重复写入 13×2、13×3 份重复数据。改动知识文档后用 `--rebuild` 显式重建。

**为什么 search() 永不抛异常、永不返回空？**
工具是 Agent 链路的一环，抛异常会让整轮对话 500；返回原始 top_k 又会把
不相关片段喂给模型诱发编造。所以三层降级：知识库没建 → 提示建库；相似度
低于阈值 → "未找到相关信息"；接口故障 → 明确的失败说明。每层都让 Agent
有话可说，而不是崩掉。

**为什么相似度阈值默认 0.25？**
向量检索的 top_k 是"矮子里拔将军"——即使知识库里完全没有相关内容，也会
返回最高分的片段。阈值把"最高分但很低分"的片段拦下来，这是"Agent 能承认
不知道"的关键机制之一（另一半是检索结果末尾附加给模型的行为提示）。
0.25 是保守起步值，接入真实 embedding 后观察分数分布再调。

**为什么工具保持同步函数（不写 async def）就够异步安全？**
LangGraph 的 `ToolNode` 对同步工具是通过 `get_executor_for_config` 拿线程池
执行器运行的：ChromaDB 查询（SQLite）和 embedding HTTP 请求的阻塞发生在
工作线程里，FastAPI 的事件循环不受影响。`test_agent.py` 的离线测试也确认了
整条链路在知识库未建/检索失败时不会炸图。

**为什么返回"相似度分数"和"来源"给模型？**
来源标注让回复可溯源（用户问"你从哪知道的"有依据）；分数是给阈值过滤和
以后调试分布用的。

---

# V5：PostgreSQL 真实持久化

V5 做的事：`query_order` / `create_ticket` 从内存假数据升级为真查 / 真写
PostgreSQL。RAG（rag.py + chroma_db）、图结构（agent.py）、接口契约
（`POST /chat` 的请求/响应结构）全部零改动。

## 架构

```
Agent（LangGraph，ainvoke 异步链路）
   │
   ├─ query_order    ──async──► SQLAlchemy ORM select ──► PostgreSQL(orders)
   ├─ create_ticket  ──async──► SQLAlchemy ORM insert  ──► PostgreSQL(tickets)
   └─ search_knowledge（同步，ToolNode 自动丢线程池）──► ChromaDB（不变）
```

## 表结构（ER）

```
orders                          tickets
─────────────────────           ────────────────────────────
order_id        PK  ◄────┐     ticket_id    PK, 自增
user_id             │    └───── order_id    FK → orders.order_id
status              │          reason      TEXT
logistics_status    │          status      DEFAULT '待处理'
logistics_detail    │          created_at  DEFAULT now()
created_at          │
```

- 一张订单可以产生多张工单（1:N），工单里的 order_id 外键指向 orders。
- 外键 `ondelete=CASCADE`：订单删除时其工单一并删除，不留下孤儿工单。
- `ticket_id` 改成了数据库自增整数（V0-V4 是内存生成的 `TK-XXXX` 字符串）：
  唯一性由数据库保证，天然有序，排查问题时可以直接按号段过滤。
- `created_at` 用 `server_default=func.now()` 由数据库生成，而不是 Python 端
  `datetime.now()`——时间基准统一在数据库，不受应用服务器时钟影响。

## 怎么跑

```bash
pip install -r requirements.txt
# .env 里配好 DATABASE_URL（见下文），并确保 PostgreSQL 服务已启动
python scripts/seed_data.py     # 灌 7 条模拟订单，幂等，可重复执行
uvicorn main:app --reload
```

验收测试（第一条改动后重新提问，回复内容应跟着变）：

```sql
UPDATE orders SET logistics_detail = '快件已到达【XX营业部】' WHERE order_id = '12345';
```

```bash
# 建工单后到库里核对：
# SELECT * FROM tickets ORDER BY ticket_id;
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "订单9527物流异常，帮我建个工单人工跟进"}'
```

## 配置（.env）

```ini
DATABASE_URL=postgresql+asyncpg://postgres:密码@localhost:5433/agent_demo
```

驱动是 `asyncpg`（SQLAlchemy 异步生态目前最成熟的 PG 驱动）。端口按本机
PostgreSQL 实际监听端口填（本机是 5433，不是默认 5432，见 postgresql.conf）。

## 每个设计决定的原因（面试可讲）

**为什么数据库层全异步（SQLAlchemy 2.0 async + asyncpg）？**
`/chat` 现在是 `async def` + `agent.ainvoke`，整条链路跑在 FastAPI 的事件循环
上。如果 DB 调用是同步阻塞的，一次查库会把整个事件循环卡住，服务在并发下
所有请求都得排队。asyncpg 的协议本身是异步的，等待数据库响应期间事件循环
可以处理别的请求。这也是 V5 唯一动了 `main.py` 的地方：`chat` 从同步改成
异步、`invoke` 换成 `ainvoke`——接口契约（路径、请求体、响应体）一字未变。

**为什么工具是 async，而 search_knowledge 保持同步也能共存？**
LangGraph 的 `ToolNode` 两条路径都有：异步图里 await 异步工具，同步工具则
自动丢线程池。所以 `query_order` / `create_ticket` 是 async（DB 操作全程
async/await），`search_knowledge` 保持同步函数不动（rag.py 是同步实现），
两者在同一张图里各走各的路径，互不阻塞。

**为什么 DB 挂了服务还能启动？**
`db.py` 的 import 绝不碰网络：`create_async_engine` 只建对象不连接。建表是
惰性的——第一次真正用到 DB 的工具调用才触发 `ensure_tables()`。PostgreSQL
没启动时，FastAPI 照常起来、`/health` 照常 200，只有调用订单/工单工具时才
返回一条明确的 `{"error": "数据库暂时不可用：..."}` 给模型组织话术，而不是
500 崩掉。连接池配了 `pool_pre_ping`（取连接前探活，数据库重启后旧连接不会
变僵尸）和 `pool_recycle`（30 分钟回收，防服务端超时踢连接）。

**为什么查不到订单返回结构化错误而不是抛异常？**
"订单不存在"是正常业务分支（用户可能输错单号），不是系统故障。返回
`{"error": ...}` 让 Agent 有话可说（引导用户核对订单号）；抛异常则会让整轮
对话直接 500。真正的系统故障（连不上库）同样以结构化错误降级——两种情况
模型拿到的都是"可以用来组织回复的信息"，而不是一个崩溃的工具。

**为什么全部走 ORM（参数化），不拼 SQL 字符串？**
订单号是用户输入直达模型的参数，拼接 SQL 就是给 SQL 注入留门。ORM 的
`session.get` / `insert` 底层全部参数化，从机制上杜绝这一类问题，也顺手
获得了跨库能力和类型安全的列映射。

**为什么 create_ticket 要先查订单再插入，而不是靠外键报错？**
外键违例抛出的 `IntegrityError` 是数据库视角的错误，用户看不懂。先查一次，
订单不存在时返回"订单不存在，无法创建工单"，Agent 能把这个原因复述给用户。
外键约束仍然保留——它是并发下的最终防线（先查后插之间订单理论上仍可能被删），
两层防护各管各的时段。

**为什么建表放在代码里（create_all）而不是单独写 SQL 迁移脚本？**
这个项目规模下，`Base.metadata.create_all`（自带 checkfirst，幂等）让"克隆
仓库 → 配 .env → 跑 seed → 起服务"四步可用，没有额外的迁移工具链。表结构
复杂化之后（改列、加索引、数据回填）再引入 Alembic 做版本化迁移，那才是
它发挥价值的时候。

**为什么 DB 建表/灌数和 ChromaDB 建库是两套逻辑？**
两个存储的生命周期完全不同：ChromaDB 的数据（embedding）计算昂贵、跟着
知识文档走，所以"建一次、显式 rebuild"；PostgreSQL 的订单数据是随时可变的
业务数据，seed 脚本只负责"确保演示数据在"，用 upsert 保证幂等，之后库里
的变化以数据库为准。

---

# V6：多轮对话记忆

V6 做的事：`/chat` 从无状态升级为基于 session 的多轮会话——同一 `session_id`
的请求共享上下文，用户可以"先报订单号，再追问细节"。图结构、工具、工具循环
零改动，记忆全部实现在编排层（main.py）+ 存储层（db.py）。

## 会话机制

```
POST /chat {message, session_id?}
      │
      ├─ session_id 缺省？→ uuid4 生成，随响应返回
      ▼
messages 表查该 session 最近 MAX_HISTORY_MESSAGES 条（滑动窗口）
      │
      ▼
历史 → HumanMessage/AIMessage 多轮数组 + 本轮用户消息 ──► agent.ainvoke
      │                                                    │
      ▼                                                    ▼
本轮 user 消息 + assistant 最终回复写回 messages 表      工具循环照常跑
```

## messages 表结构

| 列 | 类型 | 说明 |
| --- | --- | --- |
| id | INTEGER PK 自增 | 按插入顺序即时间正序，截取窗口直接按 id 排 |
| session_id | VARCHAR(64)，索引 | 会话标识 |
| role | VARCHAR(16) | 只存 user / assistant |
| content | TEXT | 消息内容 |
| created_at | TIMESTAMPTZ DEFAULT now() | 数据库生成时间 |

## 会话管理接口（附加）

```
GET    /sessions/{session_id}/history   查看某会话的历史消息
DELETE /sessions/{session_id}           清空某会话的记忆（返回删除条数）
```

## 滑动窗口策略

`MAX_HISTORY_MESSAGES=10`（.env 可配）。每次请求只携带最近 N 条历史，
截取用"倒序 LIMIT N 再反转"实现——成本恒定，会话再长也不会拖慢查询、
更不会让 prompt 无限膨胀。代价是 N 之前的上下文彻底看不见，这是本版的
刻意取舍：客服对话的有效上下文天然很短（一两单的事），先不做摘要压缩。

## 每个设计决定的原因（面试可讲）

**为什么历史以多轮 messages 数组传入，而不是拼进一段文本？**
GLM 这类对话模型对 messages 数组里角色边界的理解远好于自由文本拼接。
拼文本需要额外的分隔符约定，模型容易把历史内容当成"对自己的指令"（提示
注入的温床）；结构化数组则角色清晰、边界可靠。

**为什么只存 user 消息和 assistant 最终回复，工具调用中间过程不入库？**
恢复上下文需要的是"对话说了什么"，不是"系统怎么算出来的"。中间过程
（tool_calls / ToolMessage）依赖本轮的工具绑定关系，跨轮重放只会占 token
并可能干扰模型；且最终回复里已经包含了用户需要的结论。

**为什么截取按 id 倒序而不是 created_at？**
同一会话内消息严格按插入顺序产生，自增 id 天然就是时间序，且是主键——
按 id 排序走索引，比按 timestamptz 排序便宜，也不存在同一毫秒的排序歧义。

**为什么要在 system prompt 里加"指代消解"规则？**
光有历史还不够：模型看到历史里有现成答案时，倾向直接拿旧回复作答而不
重新查库，但物流状态是会变的。规则明确要求"代词指代的订单先确定订单号、
重新调 query_order 拿最新状态"，上下文找不到指代对象时反问用户而不是凭空
查单——这两条正好对应"多轮连贯"和"会话隔离"两条验收线。

**为什么记忆放 PostgreSQL 而不是内存？**
内存字典随 uvicorn 重启蒸发，多 worker 部署（uvicorn --workers N）时每个
进程一份记忆还互不相同。入表后重启无感、进程间共享，和 V5 的数据层复用
同一套连接池，没有额外依赖（requirements.txt 本版零新增）。

**为什么 session_id 由服务端生成并回传，而不是强制客户端传？**
老客户端（不传 session_id）完全兼容：每次请求自动获得一个新会话，行为
等同 V5 的无状态模式。想延续上下文的客户端从响应里取 session_id 带回来
即可——渐进增强，不是破坏性变更。

---

# V7：自动化评测体系

V7 做的事：建立可重复运行的评测（evaluation）体系，把"Agent 行为质量"
从主观感觉变成可量化、可对比的数字。改一行 prompt、换一个模型，跑一次
评测就知道工具选择有没有退化、边界拒绝是否还生效。

## 组成

```
evals/
├── cases.json            # 23 个评测用例（加用例只改这里，不动代码）
├── run_eval.py           # 运行器：跑用例、断言、出报告
└── results/
    ├── eval_YYYYMMDD_HHMMSS.json   # 每次运行的完整结果（留档对比用）
    └── latest_report.md            # 最新一次的 markdown 报告
```

## 用例分布（23 个，5 类）

| 类别 | 数量 | 测什么 |
| --- | --- | --- |
| tool_select 工具选择 | 6 | 该调哪个工具、参数对不对、调用顺序 |
| boundary 边界拒绝 | 4 | 不存在的订单/没单号/情绪化要求，不得乱写库 |
| memory 多轮记忆 | 4 | "它"的指代解析、上下文继承、无历史对照组 |
| rag 检索 | 4 | 命中知识库、回复含关键事实、库外不编造 |
| robustness 鲁棒性 | 5 | emoji/乱标点/超长输入/非法单号，不崩溃 |

## 怎么运行

```bash
python evals/run_eval.py            # 全量 23 个用例
python evals/run_eval.py boundary   # 只跑某一类
```

- 不起 uvicorn，直接 `build_agent()` 调用；多轮编排复用 main.py 的
  `history_to_messages` + db 的历史读写，和线上 `/chat` 同一条链路。
- 打真实 LLM（llm.py 固定 temperature=0，随机性已压到最低），不 mock。
- **评测会真实写库**：工单真建（结束后报告新建的 ticket_id 和清理 SQL），
  会话历史用 `eval-` 前缀的独立 session，跑完自动清理。
- 注意：`test_agent.py` 的离线测试也会真建一个工单，介意就定期清。

## 如何解读报告

报告按类别给出通过率，失败用例打印"期望 vs 实际"（工具序列、回复片段）。
看报告的顺序建议：先看 boundary（安全底线，必须 100%），再看 memory 和
tool_select（核心能力），最后看 robustness。通过率不是唯一指标——哪条
用例失败、失败方式是什么，比总分更有信息量（见下面的调优实录）。

## 当前基线成绩与调优实录

| 轮次 | 总分 | 结果 | 说明 |
| --- | --- | --- | --- |
| run#1（调优前基线） | 87%（20/23） | 工具选择 67%、边界 100%、记忆 75%、RAG 100%、鲁棒 100% | 暴露 3 个失败，同因：建单时机策略缺失 |
| run#2（改 prompt 后） | 100%（23/23） | 全类 100% | 3 个失败全部修复 |
| run#3 | 96%（22/23） | boundary_004 失败 | 分析结论：用例预期本身错了（见下） |
| run#4 | 96%（22/23） | tool_select_003 失败 | flaky：建单策略仍有一处模糊 |
| run#5、run#6（最终） | 100%（23/23）×2 | 全类 100% | 连续两轮全绿，稳定 |

**调优 1（针对 run#1 的 3 个失败）**：原 prompt 只说"不要随便建工单"，
导致两个方向都出错——用户只查物流时 Agent 主动建了单（越权写操作），
用户明确要求建单时 Agent 又反复犹豫不执行。把建单时机拆成明确的两条：
用户明确要求 + 订单存在 → 直接建；用户只是查询 → 只解释不建单。
改完 run#2 从 87% → 100%。

**调优 2（boundary_004 断言修正）**：run#3 中"订单88888我要直接退款"建了
工单被判失败。分析后认定是**用例预期错了**而不是 Agent 错了：用户明确
要求退款，Agent 不能自己执行退款，建工单转人工正是正确路径（政策也支持
未发货订单退款）。真正的边界是"不存在的订单/没有单号不许建单"（boundary_
001/002，始终通过）。已把该用例改为断言"先查单、再依规则行动"。

**调优 3（tool_select_003 的 flaky）**："订单12345有问题，帮我建个工单"
——数据看不出异常时，模型有时建单有时反问。明确策略写进 prompt：用户
明确要求建单且订单存在时，把用户描述记入 reason 直接建单，问题真伪由
人工判断，不要拒绝或反复追问。改后连续两轮 23/23。

## 每个设计决定的原因（面试可讲）

**为什么断言分两层：工具精确、回复宽松？**
工具调用是结构化决策，该精确断言（调了什么、参数是什么、什么顺序、
什么绝对不许调）；回复是自然语言生成，同样的问题每次措辞都不同，只能
断言"必须包含的关键事实"（如 48 小时）和"绝不许出现的内容"（如编造的
政策）。断言整句回复的评测跑一次就废。

**为什么评测不打 mock、不走 HTTP？**
mock LLM 的评测永远 100%，没有信息量——评测的意义就是度量真实模型的
行为。不走 HTTP 是为了快和稳：省掉网络序列化，跑完直接拿到工具序列
（HTTP 轨迹里只有字符串），而且编译一次 Agent 复用 23 个用例。

**为什么写库要在结束后报告工单、会话要自动清理？**
评测是高频重复操作，垃圾数据必须自愈：会话历史（可再生的测试产物）
跑完即删；工单是"真实业务写入"，报告出来让操作者自己决定删不删——
分寸感和 V5 建库幂等、V6 会话清理是同一个原则：测试不留下需要人肉
善后的现场。

**为什么保存每次运行的 JSON？**
评测的价值在趋势不在单次。results/ 下的历史文件让你能回答"我改了
prompt 之后工具选择有没有退化"这类问题——这正是 V7 要解决的原始问题。



