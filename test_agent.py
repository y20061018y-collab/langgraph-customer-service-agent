"""
不需要真实 API Key 也能验证这个 Agent 的图结构对不对。

思路：用一个"假LLM"，让它按脚本依次返回：
    第1次调用 -> 决定调用 query_order
    第2次调用 -> 看到订单异常，决定调用 search_knowledge
    第3次调用 -> 决定调用 create_ticket
    第4次调用 -> 给出最终自然语言回复（不再调用任何工具）

如果整个 StateGraph 搭对了，跑完这4轮之后应该正确停在 END，
并且 main.py 里的轨迹组装逻辑能正确把这3次工具调用还原出来。

这不是在测试"LLM聪不聪明"，而是在测试"图的状态流转对不对"——
这才是面试会问到的部分，也是真正接入真实模型之前最值得先跑通的东西。
"""
import asyncio
import uuid

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_agent


class FakeToolCallingLLM(FakeMessagesListChatModel):
    """
    真实的 ChatOpenAI 会把 tools 的 schema 发给模型，让模型自己决定调不调、调哪个。
    这里我们不需要真的做这个决策过程（脚本已经写死了顺序），
    只需要让 .bind_tools() 这个接口存在、不报错即可，所以直接返回自身。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def make_scripted_llm():
    scripted_responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_order", "args": {"order_id": "9527"}, "id": "call_1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge", "args": {"query": "物流长时间不更新"}, "id": "call_2"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "create_ticket", "args": {"order_id": "9527", "reason": "物流异常超48小时未更新"}, "id": "call_3"}],
        ),
        AIMessage(
            content="已为您查询订单9527，物流在广州转运中心滞留超48小时，判定为物流异常，已为您创建工单 TK-XXXXXXXX，客服会尽快跟进处理。",
        ),
    ]
    return FakeToolCallingLLM(responses=scripted_responses)


def run_async(coro):
    """在独立事件循环里跑一个协程，并在**同一个循环**里丢弃连接池。

    asyncpg 连接绑定在创建它的循环上，而 asyncio.run() 每次都开新循环；
    生产环境 uvicorn 只有一个常驻循环不受影响，但测试进程里连续多个
    asyncio.run 会让连接池里的旧连接失效。所以每次循环结束前显式 dispose，
    下个循环重建连接。

    dispose 必须在跑协程的同一个循环里、循环关闭之前 await：如果放在另一个
    asyncio.run 里，关闭旧连接时会触发 "attached to a different loop" /
    "Event loop is closed"，打印 "Exception closing connection" 堆栈。
    """
    import db

    async def _wrapped():
        try:
            return await coro
        finally:
            await db.engine.dispose()

    return asyncio.run(_wrapped())


def test_full_loop():
    fake_llm = make_scripted_llm()
    agent = build_agent(llm=fake_llm)

    # V5：工具改成 async（真查/真写 PostgreSQL）后，整条链路走 ainvoke，
    # 和 main.py 的生产路径保持一致（同步 invoke 已无法执行 async 工具）。
    result = run_async(
        agent.ainvoke(
            {"messages": [HumanMessage(content="我的订单9527为什么还没发货，如果物流异常帮我创建工单")]}
        )
    )

    messages = result["messages"]

    # 断言1：最后一条消息是没有 tool_calls 的最终回复
    final = messages[-1]
    assert isinstance(final, AIMessage)
    assert not final.tool_calls
    assert "工单" in final.content
    print("✅ 最终回复正确生成:", final.content)

    # 断言2：三个工具确实都被依次调用了
    from langchain_core.messages import ToolMessage
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3
    print(f"✅ 共执行了 {len(tool_messages)} 次工具调用，依次为:")
    for tm in tool_messages:
        print(f"   - {tm.name}: {tm.content[:80]}")

    print("\n🎉 最小闭环跑通：agent → tools → agent → tools → agent → tools → agent(结束)")


# ---------- V6：会话记忆 ----------

def test_history_window_and_order():
    """滑动窗口：limit 生效、截取的是"最近"的 N 条、且按时间正序返回。"""
    import asyncio
    import db

    async def run():
        await db.ensure_tables()
        sid = f"test-v6-window-{uuid.uuid4().hex[:8]}"
        try:
            for i in range(12):
                role = "user" if i % 2 == 0 else "assistant"
                await db.save_message(sid, role, f"消息{i}")
            rows = await db.load_history(sid, limit=3)
            return sid, [r.content for r in rows]
        finally:
            await db.clear_session(sid)

    sid, contents = run_async(run())
    assert contents == ["消息9", "消息10", "消息11"], contents
    print("✅ 滑动窗口生效：12 条历史只取最近 3 条，且按时间正序:", contents)


def test_history_to_messages():
    """历史行 → LangChain 多轮消息数组的角色映射正确。"""
    from main import history_to_messages
    from langchain_core.messages import HumanMessage, AIMessage

    class Row:
        def __init__(self, role, content):
            self.role, self.content = role, content

    msgs = history_to_messages(
        [Row("user", "查订单12345"), Row("assistant", "好的"), Row("user", "有异常吗")]
    )
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage, HumanMessage]
    assert msgs[0].content == "查订单12345"
    print("✅ 历史消息正确映射为多轮 messages 数组（user→Human, assistant→AI）")


# ---------- 检索轨迹解析（retrieval_trace.py）：5 种 status 各一个测试 ----------

HIT_OUTPUT = (
    "从知识库中检索到以下相关片段：\n\n"
    "[1]（来源: 11-发票政策，相似度 0.62）\n发票政策\n支持电子普通发票和增值税专用发票。\n\n"
    "[2]（来源: 03-退款流程与时效，相似度 0.31）\n退款流程\n仅退款适用于未收到货。"
    "\n\n（如果以上片段与用户问题无关，请直接告知用户知识库中没有相关内容，不要编造。）"
)


def test_retrieval_constants_match_rag():
    """retrieval_trace 里重复的降级文案必须与 rag.py 一致，否则解析会漂移。"""
    import rag
    import retrieval_trace as rt

    assert rt.NOT_FOUND_MSG == rag.NOT_FOUND_MSG
    assert rt.KB_NOT_BUILT_MSG == rag.KB_NOT_BUILT_MSG
    # rag.search 的异常分支以 "知识库检索失败：" 开头（见 rag.py 最后一行 return）
    assert rt.ERROR_PREFIX == "知识库检索失败"
    print("✅ retrieval_trace 的降级文案与 rag.py 一致")


def test_parse_status_hit():
    from retrieval_trace import parse_retrieval

    r = parse_retrieval("怎么开发票", HIT_OUTPUT)
    assert (r.status, r.degraded, r.query) == ("hit", False, "怎么开发票")
    assert [h.source for h in r.hits] == ["11-发票政策", "03-退款流程与时效"]
    assert [h.similarity for h in r.hits] == [0.62, 0.31]
    assert r.hits[0].snippet == "发票政策\n支持电子普通发票和增值税专用发票。"
    assert "如果以上片段" not in r.hits[1].snippet
    print("✅ status=hit：来源、相似度、片段（含多行）解析正确")


def test_parse_status_not_found():
    import rag
    from retrieval_trace import parse_retrieval

    r = parse_retrieval("q", rag.NOT_FOUND_MSG)
    assert (r.status, r.degraded, r.hits) == ("not_found", True, [])
    print("✅ status=not_found")


def test_parse_status_kb_not_built():
    import rag
    from retrieval_trace import parse_retrieval

    r = parse_retrieval("q", rag.KB_NOT_BUILT_MSG)
    assert (r.status, r.degraded, r.hits) == ("kb_not_built", True, [])
    print("✅ status=kb_not_built")


def test_parse_status_error():
    from retrieval_trace import parse_retrieval

    out = "知识库检索失败：[SSL: UNEXPECTED_EOF_WHILE_READING] EOF。请告知用户暂时无法查询知识库。"
    r = parse_retrieval("q", out)
    assert (r.status, r.degraded, r.hits) == ("error", True, [])
    print("✅ status=error（embedding / 网络异常）")


def test_parse_status_unknown():
    """格式异常：识别不出来记为 unknown，不能默认当 hit，也不下"是否降级"的结论。"""
    from retrieval_trace import parse_retrieval

    for bad in [
        "",
        "一段完全无关的文字",
        "从知识库中检索到以下相关片段：\n\n（没有任何片段头部）",
        "[1]（来源: x，相似度 1.2.3）\n内容",  # 分数无法转成 float
    ]:
        r = parse_retrieval("q", bad)
        assert (r.status, r.degraded, r.hits) == ("unknown", None, []), (bad, r)
    print("✅ status=unknown：格式异常不会被当作 hit")


def test_parse_real_rag_search_output():
    """用真实的 rag.search 产生输出，再交给 parse_retrieval：rag.py 的文案或片段格式一变，这里会失败。

    只 monkeypatch 外部依赖（embedding 接口、ChromaDB 集合），rag.search 本身是真实代码。
    """
    import rag
    from retrieval_trace import parse_retrieval

    class FakeCollection:
        # 与当前配置一致的建库元数据（维度 4 对应下面假 embedding 返回的向量长度）
        metadata = {
            "embedding_provider": rag.CONFIG.provider,
            "embedding_model": rag.CONFIG.model,
            "embedding_dimension": 4,
        }

        def __init__(self, distances):
            self.distances = distances

        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "documents": [["发票政策\n支持电子普通发票。", "退款流程\n仅退款适用于未收到货。"]],
                "metadatas": [[{"source": "11-发票政策"}, {"source": "03-退款流程与时效"}]],
                "distances": [self.distances],
            }

    saved = (rag.embed_texts, rag.get_collection)
    try:
        # error：embedding 接口抛异常，rag.search 走真实的 except 分支
        def boom(texts):
            raise RuntimeError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF")

        rag.embed_texts = boom
        rag.get_collection = lambda: FakeCollection([0.2, 0.5])
        out = rag.search("怎么开发票")
        r = parse_retrieval("怎么开发票", out)
        assert (r.status, r.degraded, r.hits) == ("error", True, []), (out, r)

        # hit：相似度 0.8 / 0.5（距离 0.2 / 0.5），都高于阈值
        rag.embed_texts = lambda texts: [[0.0] * 4 for _ in texts]
        out = rag.search("怎么开发票")
        r = parse_retrieval("怎么开发票", out)
        assert (r.status, r.degraded) == ("hit", False), (out, r)
        assert [h.source for h in r.hits] == ["11-发票政策", "03-退款流程与时效"]
        assert [h.similarity for h in r.hits] == [0.8, 0.5]
        assert r.hits[0].snippet == "发票政策\n支持电子普通发票。"
        assert "如果以上片段" not in r.hits[1].snippet

        # not_found：全部低于阈值（距离 0.9 -> 相似度 0.1 < 0.25）
        rag.get_collection = lambda: FakeCollection([0.9, 0.95])
        r = parse_retrieval("q", rag.search("q"))
        assert (r.status, r.degraded, r.hits) == ("not_found", True, [])
    finally:
        rag.embed_texts, rag.get_collection = saved
    print("✅ 真实 rag.search 的输出经 parse_retrieval：error / hit / not_found 判定正确")


def test_eval_extract_tool_records():
    """run_eval 留档：工具返回只存前 500 字符，结构化 retrievals 全存，且只含 search_knowledge。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
    import run_eval
    from langchain_core.messages import ToolMessage

    long_hit = HIT_OUTPUT + "x" * 1000
    result = {
        "messages": [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[
                {"name": "query_order", "args": {"order_id": "1"}, "id": "a"},
                {"name": "search_knowledge", "args": {"query": "发票"}, "id": "b"},
            ]),
            ToolMessage(content='{"error": "x"}', tool_call_id="a", name="query_order"),
            ToolMessage(content=long_hit, tool_call_id="b", name="search_knowledge"),
            AIMessage(content="done"),
        ]
    }
    outputs, retrievals = run_eval.extract_tool_records(result)
    assert [o["name"] for o in outputs] == ["query_order", "search_knowledge"]
    assert len(outputs[1]["output"]) == 500 and outputs[1]["truncated"] is True
    assert outputs[1]["output_chars"] == len(long_hit)
    assert outputs[0]["truncated"] is False
    assert len(retrievals) == 1 and retrievals[0]["status"] == "hit"
    assert [h["source"] for h in retrievals[0]["hits"]] == ["11-发票政策", "03-退款流程与时效"]
    counts = run_eval.retrieval_status_counts(retrievals)
    assert counts == {"hit": 1, "not_found": 0, "kb_not_built": 0, "error": 0, "unknown": 0}
    print("✅ run_eval 留档：返回原文截断到 500 字符，retrievals 完整")


def test_eval_llm_error_does_not_abort_run():
    """对话模型调用抛异常：该用例记为调用异常（不通过、不算模型判断失败），整轮继续，结果文件照常写出。

    用假的 agent（ainvoke 对含 BOOM 的消息抛异常），不调用任何真实接口；
    结果目录指向临时目录，不污染 evals/results。
    """
    import json
    import sys, os
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
    import run_eval

    class FakeAgent:
        async def ainvoke(self, state):
            if any("BOOM" in str(m.content) for m in state["messages"]):
                raise ConnectionError("fake LLM down")
            return {"messages": [*state["messages"], AIMessage(content="好的")]}

    cases = {"cases": [
        {"id": "t_err", "category": "robustness", "description": "模型报错",
         "messages": [{"role": "user", "content": "BOOM"}], "expect": {"reply_contains": ["好的"]}},
        {"id": "t_ok", "category": "robustness", "description": "正常",
         "messages": [{"role": "user", "content": "你好"}], "expect": {"reply_contains": ["好的"]}},
    ]}

    # 单个用例：异常被捕获成记录，而不是向上抛
    rec = run_async(run_eval.run_case(FakeAgent(), cases["cases"][0], "unit"))
    assert rec["passed"] is False and rec["failures"] == []
    assert rec["llm_error"]["type"] == "ConnectionError" and "fake LLM down" in rec["llm_error"]["message"]
    assert rec["llm_error"]["turn"] == 1
    run_async(db_clear("eval-unit-t_err"))

    # 整轮：走 main()，第二个用例仍然跑完并通过
    tmp = tempfile.mkdtemp()
    cases_path = os.path.join(tmp, "llm_err_cases.json")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    old = (run_eval.RESULTS_DIR, run_eval.build_agent, sys.argv)
    run_eval.RESULTS_DIR = os.path.join(tmp, "results")
    run_eval.build_agent = lambda: FakeAgent()
    sys.argv = ["run_eval.py", "--cases", cases_path]
    try:
        run_async(run_eval.main())
    finally:
        run_eval.RESULTS_DIR, run_eval.build_agent, sys.argv = old

    files = os.listdir(os.path.join(tmp, "results"))
    result_file = [x for x in files if x.endswith(".json")][0]
    assert result_file.startswith("llm_err_cases_")
    data = json.load(open(os.path.join(tmp, "results", result_file), encoding="utf-8"))
    by_id = {r["id"]: r for r in data["results"]}
    assert by_id["t_err"]["passed"] is False and by_id["t_err"]["llm_error"]["type"] == "ConnectionError"
    assert by_id["t_ok"]["passed"] is True and "llm_error" not in by_id["t_ok"]
    assert (data["meta"]["total"], data["meta"]["passed"]) == (2, 1)
    assert data["meta"]["llm_error_ids"] == ["t_err"]
    import rag
    assert data["meta"]["embedding_model"] == rag.EMBEDDING_MODEL_NAME
    assert "API_KEY" not in json.dumps(data["meta"])
    report = open(os.path.join(tmp, "results", "llm_err_cases_latest_report.md"), encoding="utf-8").read()
    assert "对话模型调用异常" in report and "t_err" in report
    assert "## 失败用例详情" not in report  # 调用异常不算模型判断失败
    print("✅ run_eval：对话模型异常记为调用异常，整轮继续，meta 记录 embedding 模型名")


async def db_clear(session_id):
    import db
    await db.clear_session(session_id)


def test_chat_endpoint_and_index():
    """POST /chat 响应带 retrievals 新字段（旧字段不变），GET / 返回演示页。"""
    import httpx

    import db
    import main

    main._agent = build_agent(llm=make_scripted_llm())

    async def run():
        transport = httpx.ASGITransport(app=main.app)
        sid = f"test-demo-{uuid.uuid4().hex[:8]}"
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                chat = await c.post("/chat", json={"message": "查订单9527", "session_id": sid})
                index = await c.get("/")
            return chat, index
        finally:
            await db.clear_session(sid)
            main._agent = None

    chat, index = run_async(run())
    body = chat.json()
    assert chat.status_code == 200, chat.text
    assert [t["tool_name"] for t in body["tool_calls"]] == [
        "query_order", "search_knowledge", "create_ticket",
    ]
    # 测试环境没有建知识库，search_knowledge 走"知识库尚未建立"降级
    assert len(body["retrievals"]) == 1
    assert body["retrievals"][0]["query"] == "物流长时间不更新"
    assert body["retrievals"][0]["degraded"] is True
    assert index.status_code == 200 and "text/html" in index.headers["content-type"]
    print("✅ /chat 返回 retrievals（降级被标出），GET / 返回演示页")


# ---------- embedding 提供方 / 向量库一致性 / 重试（rag.py）：不调用任何真实接口 ----------

class _ListHandler:
    """把 rag logger 的日志收集到列表里，便于断言（也顺便不让日志刷屏）。"""

    def __init__(self):
        import logging

        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.records.append(record)
        self.logger = logging.getLogger("rag")

    def __enter__(self):
        import logging

        self.old = (self.logger.handlers[:], self.logger.level, self.logger.propagate)
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        return self

    def __exit__(self, *exc):
        self.logger.handlers, level, self.logger.propagate = self.old
        self.logger.setLevel(level)

    def text(self, levelname=None):
        return "\n".join(r.getMessage() for r in self.records if levelname in (None, r.levelname))


class _FakeIndex:
    """假的 ChromaDB 集合：只实现一致性校验 / 检索 / 建库会用到的方法。"""

    def __init__(self, metadata, dim=1024, count=13):
        self.metadata = metadata
        self.dim = dim
        self._count = count

    def count(self):
        return self._count

    def get(self, limit, include):
        return {"embeddings": [[0.0] * self.dim] if self.dim else []}

    def query(self, **kwargs):
        return {
            "documents": [["发票政策\n支持电子普通发票。"]],
            "metadatas": [[{"source": "11-发票政策"}]],
            "distances": [[0.2]],
        }


def _meta(provider, model, dim):
    return {"hnsw:space": "cosine", "embedding_provider": provider, "embedding_model": model, "embedding_dimension": dim}


class _RagPatch:
    """临时替换 rag 的模块级状态，退出时恢复。"""

    NAMES = ("CONFIG", "CONFIG_ERROR", "embed_texts", "get_collection", "_TRANSPORT", "_sleep",
             "_legacy_warned", "CHROMA_DIR", "_collection_cache")

    def __init__(self, **overrides):
        self.overrides = overrides

    def __enter__(self):
        import rag

        self.rag = rag
        self.saved = {n: getattr(rag, n) for n in self.NAMES}
        for k, v in self.overrides.items():
            setattr(rag, k, v)
        return rag

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.rag, k, v)


def test_embedding_config_parsing():
    """配置解析：默认值与原来一致；ollama 的默认值；非法取值不让 import 失败而是报配置错误。"""
    import rag

    c = rag.load_embedding_config({})
    assert (c.provider, c.model, c.base_url, c.api_key) == ("zhipu", "embedding-3", "", None)
    assert (c.dimensions, c.similarity_threshold, c.timeout) == (1024, 0.25, 60.0)
    assert c.describe() == "zhipu/embedding-3/1024"

    # zhipu：没配 EMBEDDING_* 时回退到 LLM_*（与原来一致），EMBEDDING_* 优先
    c = rag.load_embedding_config({"LLM_BASE_URL": "https://llm.example/v4/", "LLM_API_KEY": "sk-llm"})
    assert (c.base_url, c.api_key) == ("https://llm.example/v4", "sk-llm")
    c = rag.load_embedding_config({
        "LLM_BASE_URL": "https://llm.example/v4/", "LLM_API_KEY": "sk-llm",
        "EMBEDDING_BASE_URL": "https://emb.example/v4/", "EMBEDDING_API_KEY": "sk-emb",
    })
    assert (c.base_url, c.api_key) == ("https://emb.example/v4", "sk-emb")

    # ollama：默认 bge-m3 + 本机地址；不回退到 LLM_*，也不带 key
    c = rag.load_embedding_config({"EMBEDDING_PROVIDER": " Ollama ", "LLM_BASE_URL": "https://llm/", "LLM_API_KEY": "sk"})
    assert (c.provider, c.model, c.base_url, c.api_key) == ("ollama", "bge-m3", "http://localhost:11434", None)
    assert c.describe() == "ollama/bge-m3/?"
    c = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_BASE_URL": "http://host.docker.internal:11434/"})
    assert c.base_url == "http://host.docker.internal:11434"

    # 阈值沿用旧变量名 EMBEDDING_SIMILARITY_THRESHOLD；SIMILARITY_THRESHOLD 不读
    assert rag.load_embedding_config({"EMBEDDING_SIMILARITY_THRESHOLD": "0.6"}).similarity_threshold == 0.6
    assert rag.load_embedding_config({"SIMILARITY_THRESHOLD": "0.6"}).similarity_threshold == 0.25
    assert rag.load_embedding_config({"EMBEDDING_TIMEOUT": "5"}).timeout == 5.0

    for bad in [{"EMBEDDING_PROVIDER": "openai"}, {"EMBEDDING_DIMENSIONS": "abc"},
                {"EMBEDDING_SIMILARITY_THRESHOLD": "x"}, {"EMBEDDING_TIMEOUT": "soon"}]:
        try:
            rag.load_embedding_config(bad)
        except rag.EmbeddingConfigError:
            continue
        raise AssertionError(f"应当报配置错误：{bad}")

    # repr 不带 key
    assert "sk-emb-SECRET" not in repr(rag.load_embedding_config({"EMBEDDING_API_KEY": "sk-emb-SECRET"}))
    print("✅ embedding 配置解析：默认值不变、ollama 默认值、非法取值报配置错误、repr 不含 key")


def test_check_index_consistency():
    """一致性校验：provider / model / 维度不一致都报出来；无元数据的旧库只在默认配置 + 1024 维时放行。"""
    import rag

    zhipu = rag.load_embedding_config({})
    ollama = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama"})

    assert rag.check_index(_FakeIndex(_meta("zhipu", "embedding-3", 1024)), zhipu) is None
    assert rag.check_index(_FakeIndex(_meta("zhipu", "embedding-3", 1024)), zhipu, query_dim=1024) is None
    p = rag.check_index(_FakeIndex(_meta("zhipu", "embedding-3", 1024)), zhipu, query_dim=512)
    assert p and "库：zhipu/embedding-3/1024" in p and "当前：zhipu/embedding-3/512" in p and "--rebuild" in p

    p = rag.check_index(_FakeIndex(_meta("zhipu", "embedding-3", 1024)), ollama)
    assert p and "库：zhipu/embedding-3/1024" in p and "当前：ollama/bge-m3/?" in p
    p = rag.check_index(_FakeIndex(_meta("zhipu", "embedding-2", 1024)), zhipu)
    assert p and "embedding-2" in p
    assert rag.check_index(_FakeIndex(_meta("ollama", "bge-m3", 1024)), ollama, query_dim=1024) is None

    legacy = {"hnsw:space": "cosine"}
    with _RagPatch(_legacy_warned=False), _ListHandler() as logs:
        assert rag.check_index(_FakeIndex(legacy, dim=1024), zhipu) is None
        assert rag.check_index(_FakeIndex(legacy, dim=1024), zhipu, query_dim=1024) is None
        assert len([r for r in logs.records if r.levelname == "WARNING"]) == 1  # 只告警一次
        assert "--rebuild" in logs.text("WARNING")
        for idx, cfg in [
            (_FakeIndex(legacy, dim=512), zhipu),  # 旧库但维度不是 1024
            (_FakeIndex(legacy, dim=1024), ollama),  # 非默认 provider
            (_FakeIndex(legacy, dim=1024), rag.load_embedding_config({"EMBEDDING_DIMENSIONS": "2048"})),
            (_FakeIndex(legacy, dim=1024), rag.load_embedding_config({"EMBEDDING_MODEL_NAME": "embedding-2"})),
        ]:
            p = rag.check_index(idx, cfg)
            assert p and "旧版建的库" in p and "--rebuild" in p, (cfg, p)
    print("✅ 向量库一致性：provider / model / 维度不一致明确报错，旧库只在默认配置下放行并告警一次")


def test_search_returns_error_on_index_mismatch():
    """search 永不抛异常：不一致 / 配置错误时返回"知识库检索失败："开头的文案，parse_retrieval 判为 error。"""
    import rag
    from retrieval_trace import parse_retrieval

    called = []

    def must_not_embed(texts):
        called.append(texts)
        return [[0.0] * 1024 for _ in texts]

    ollama = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama"})
    # provider 不一致：在调用 embedding 之前就失败
    with _RagPatch(CONFIG=ollama, embed_texts=must_not_embed,
                   get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))), _ListHandler() as logs:
        out = rag.search("怎么开发票")
    assert out.startswith("知识库检索失败：向量库与当前 embedding 配置不一致"), out
    assert "--rebuild" in out and out.endswith("请告知用户暂时无法查询知识库。")
    assert called == []
    assert parse_retrieval("q", out).status == "error"
    assert "不一致" in logs.text("ERROR")

    # 维度不一致：拿到查询向量后才能发现
    with _RagPatch(CONFIG=ollama, embed_texts=lambda texts: [[0.0] * 768 for _ in texts],
                   get_collection=lambda: _FakeIndex(_meta("ollama", "bge-m3", 1024))), _ListHandler():
        out = rag.search("怎么开发票")
    assert out.startswith("知识库检索失败：") and "当前：ollama/bge-m3/768" in out, out
    assert parse_retrieval("q", out).status == "error"

    # 配置错误（例如 EMBEDDING_PROVIDER 拼错）
    with _RagPatch(CONFIG=None, CONFIG_ERROR="EMBEDDING_PROVIDER 只能是 zhipu/ollama，当前为 'openai'",
                   get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))), _ListHandler():
        out = rag.search("怎么开发票")
    assert out.startswith("知识库检索失败：embedding 配置错误") and parse_retrieval("q", out).status == "error"

    # 一致时照常命中（默认 zhipu 配置）
    zhipu = rag.load_embedding_config({})
    with _RagPatch(CONFIG=zhipu, embed_texts=lambda texts: [[0.0] * 1024 for _ in texts],
                   get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))):
        r = parse_retrieval("q", rag.search("怎么开发票"))
    assert r.status == "hit" and r.hits[0].similarity == 0.8
    print("✅ search：不一致 / 配置错误返回『知识库检索失败：』（判为 error），provider 不一致时不调用 embedding")


def test_build_refuses_mismatched_index():
    """build_kb：已有数据但与当前配置不一致时，不跳过、不追加，要求 --rebuild；一致时照旧跳过。"""
    import rag

    called = []
    ollama = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama"})
    with _RagPatch(CONFIG=ollama, embed_texts=lambda texts, stage="build": called.append(1),
                   get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))):
        try:
            rag.build_knowledge_base()
        except rag.IndexMismatchError as e:
            assert "--rebuild" in str(e) and "13 个 chunk" in str(e)
        else:
            raise AssertionError("不一致时应当拒绝")
        assert called == []

    with _RagPatch(CONFIG=ollama, get_collection=lambda: _FakeIndex(_meta("ollama", "bge-m3", 1024))):
        stats = rag.build_knowledge_base()
    assert stats["skipped"] is True and stats["chunks"] == 13
    print("✅ build_kb：不一致的库拒绝继续（需 --rebuild），一致时跳过")


def test_build_and_search_with_real_chroma():
    """真实 ChromaDB（临时目录）+ 假 embedding：建库写入元数据、读回校验、切换配置后失败、--rebuild 后恢复。"""
    import tempfile

    import rag
    from retrieval_trace import parse_retrieval

    def fake_embed(texts, stage="search"):
        # 确定性的 3 维向量；查询"发票"与第 11 篇（发票政策）方向最接近
        return [[1.0, 0.0, 0.1] if ("发票" in t) else [0.0, 1.0, 0.1] for t in texts]

    zhipu = rag.load_embedding_config({})
    ollama = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama"})
    with _RagPatch(CHROMA_DIR=tempfile.mkdtemp(), _collection_cache=None, CONFIG=zhipu, embed_texts=fake_embed,
                   get_collection=rag.get_collection), _ListHandler() as logs:
        stats = rag.build_knowledge_base()
        assert stats["skipped"] is False and stats["chunks"] == 13
        meta = rag.get_collection().metadata
        assert meta == _meta("zhipu", "embedding-3", 3), meta
        assert rag.index_info() == {"embedding_provider": "zhipu", "embedding_model": "embedding-3", "embedding_dimension": 3}
        assert rag.log_index_status() == "ok"

        r = parse_retrieval("q", rag.search("开发票"))
        assert r.status == "hit" and "11-发票政策" in [h.source for h in r.hits], r

        # 再跑一次：一致 → 跳过
        assert rag.build_knowledge_base()["skipped"] is True

        # 换成 ollama 配置但不重建：检索、建库、启动日志都明确失败
        rag.CONFIG = ollama
        assert parse_retrieval("q", rag.search("开发票")).status == "error"
        assert rag.log_index_status() == "mismatch"
        try:
            rag.build_knowledge_base()
        except rag.IndexMismatchError:
            pass
        else:
            raise AssertionError("应当要求 --rebuild")

        # 重建失败（embedding 抛异常）时旧库原样保留
        def broken(texts, stage="search"):
            raise RuntimeError("embedding down")
        rag.embed_texts = broken
        try:
            rag.build_knowledge_base(rebuild=True)
        except RuntimeError:
            pass
        assert rag.get_collection().count() == 13 and rag.get_collection().metadata["embedding_provider"] == "zhipu"

        # --rebuild 成功后元数据换成 ollama
        rag.embed_texts = fake_embed
        rag.build_knowledge_base(rebuild=True)
        assert rag.get_collection().metadata == _meta("ollama", "bge-m3", 3)
        assert parse_retrieval("q", rag.search("开发票")).status == "hit"
        assert "不一致" in logs.text("ERROR")
    print("✅ 真实 ChromaDB：建库写入元数据并读回，换配置不重建时失败，重建失败不删旧库，--rebuild 后恢复")


def test_log_index_status():
    import rag

    zhipu = rag.load_embedding_config({})
    with _ListHandler() as logs:
        with _RagPatch(CONFIG=zhipu, get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))):
            assert rag.log_index_status() == "ok"
        with _RagPatch(CONFIG=zhipu, _legacy_warned=False, get_collection=lambda: _FakeIndex({"hnsw:space": "cosine"})):
            assert rag.log_index_status() == "legacy"
        with _RagPatch(CONFIG=zhipu, get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024), count=0)):
            assert rag.log_index_status() == "kb_not_built"
        with _RagPatch(CONFIG=zhipu, get_collection=lambda: _FakeIndex(_meta("ollama", "bge-m3", 1024))):
            assert rag.log_index_status() == "mismatch"
        with _RagPatch(CONFIG=None, CONFIG_ERROR="bad"):
            assert rag.log_index_status() == "config_error"
    print("✅ 启动日志：ok / legacy / kb_not_built / mismatch / config_error")


def test_embedding_retry_and_requests():
    """重试：连接类错误最多 3 次（等 1s、3s），读超时和 HTTP 错误不重试；请求格式按 provider 区分；日志不含 key。"""
    import httpx

    import rag
    from retrieval_trace import parse_retrieval

    SECRET = "sk-test-SECRET-123"
    zhipu = rag.load_embedding_config({"EMBEDDING_BASE_URL": "https://emb.example/v4/", "EMBEDDING_API_KEY": SECRET})
    ollama = rag.load_embedding_config({"EMBEDDING_PROVIDER": "ollama"})

    def run(cfg, script, texts):
        """script：每次请求的动作，异常类（会被构造并抛出）或 (status, json)。返回 (结果或异常, 请求列表, sleeps)。"""
        requests, sleeps = [], []

        def handler(request):
            requests.append(request)
            action = script[len(requests) - 1]
            if isinstance(action, type):
                raise action("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF", request=request)
            return httpx.Response(action[0], json=action[1])

        with _RagPatch(CONFIG=cfg, _TRANSPORT=httpx.MockTransport(handler), _sleep=sleeps.append):
            try:
                out = rag.embed_texts(texts)
            except Exception as e:  # noqa: BLE001
                out = e
        return out, requests, sleeps

    ok_zhipu = (200, {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]})

    with _ListHandler() as logs:
        # 两次连接类错误后成功：按 index 排回原顺序，等待 1s、3s
        out, reqs, sleeps = run(zhipu, [httpx.ConnectError, httpx.ReadError, ok_zhipu], ["a", "b"])
        assert out == [[1.0, 0.0], [0.0, 1.0]], out
        assert len(reqs) == 3 and sleeps == [1.0, 3.0]
        assert str(reqs[0].url) == "https://emb.example/v4/embeddings"
        assert reqs[0].headers["Authorization"] == f"Bearer {SECRET}"
        assert json_body(reqs[0]) == {"model": "embedding-3", "input": ["a", "b"], "dimensions": 1024}
        warn = logs.text("WARNING")
        assert "ConnectError" in warn and "ReadError" in warn and "stage=search" in warn and "第 1/3 次" in warn

        # 连续 3 次 ReadError：放弃并抛出原异常
        out, reqs, sleeps = run(zhipu, [httpx.ReadError] * 3, ["a"])
        assert isinstance(out, httpx.ReadError) and len(reqs) == 3 and sleeps == [1.0, 3.0]
        assert "重试次数用尽" in logs.text("ERROR")

        # 读超时不重试
        out, reqs, sleeps = run(zhipu, [httpx.ReadTimeout, ok_zhipu], ["a", "b"])
        assert isinstance(out, httpx.ReadTimeout) and len(reqs) == 1 and sleeps == []
        assert "超时类错误不重试" in logs.text("ERROR")
        # 连接超时属于连接类错误，重试
        out, reqs, sleeps = run(zhipu, [httpx.ConnectTimeout, ok_zhipu], ["a", "b"])
        assert out == [[1.0, 0.0], [0.0, 1.0]] and sleeps == [1.0]

        # HTTP 4xx / 5xx 不重试，报错文案沿用原来的格式
        for status in (401, 404, 500, 503):
            out, reqs, sleeps = run(zhipu, [(status, {"error": "x"}), ok_zhipu], ["a"])
            assert isinstance(out, RuntimeError) and f"embedding 接口返回 {status}" in str(out), out
            assert len(reqs) == 1 and sleeps == []

        # ollama：/api/embed，不带 Authorization、不发 dimensions；17 条分两批
        texts = [f"t{i}" for i in range(17)]
        out, reqs, sleeps = run(ollama, [
            (200, {"model": "bge-m3", "embeddings": [[float(i), 1.0] for i in range(16)]}),
            (200, {"model": "bge-m3", "embeddings": [[16.0, 1.0]]}),
        ], texts)
        assert out == [[float(i), 1.0] for i in range(17)], out
        assert [str(r.url) for r in reqs] == ["http://localhost:11434/api/embed"] * 2
        assert "authorization" not in reqs[0].headers
        assert json_body(reqs[0]) == {"model": "bge-m3", "input": texts[:16]}
        assert json_body(reqs[1]) == {"model": "bge-m3", "input": texts[16:]}

        # ollama 返回条数不对 → 明确报错
        out, _, _ = run(ollama, [(200, {"embeddings": [[1.0]]})], ["a", "b"])
        assert isinstance(out, RuntimeError) and "期望 2 条" in str(out)

        # zhipu 缺 key：不发请求，报原来的错误
        out, reqs, _ = run(rag.load_embedding_config({"EMBEDDING_BASE_URL": "https://emb/"}), [ok_zhipu], ["a"])
        assert isinstance(out, RuntimeError) and "缺少 embedding 配置" in str(out) and reqs == []

        # 经过 search：重试用尽后仍然返回"知识库检索失败："，判为 error
        requests = []

        def always_eof(request):
            requests.append(request)
            raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF", request=request)

        with _RagPatch(CONFIG=zhipu, _TRANSPORT=httpx.MockTransport(always_eof), _sleep=lambda s: None,
                       get_collection=lambda: _FakeIndex(_meta("zhipu", "embedding-3", 1024))):
            out = rag.search("怎么开发票")
        assert out.startswith("知识库检索失败：[SSL: UNEXPECTED_EOF_WHILE_READING]"), out
        assert parse_retrieval("q", out).status == "error" and len(requests) == 3

        all_logs = logs.text()
        assert SECRET not in all_logs and "Bearer" not in all_logs
    print("✅ embedding 重试：连接类错误重试 3 次（1s/3s），读超时与 HTTP 错误不重试，请求格式正确，日志不含 key")


def json_body(request):
    import json

    return json.loads(request.content)


def test_threshold_probe_rule():
    """阈值选取规则（事先写定）：可分取中点，不可分取 min A 向下取整；过滤率按业务相近 / 无关分列。"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
    import threshold_probe as tp

    c = tp.choose_threshold([0.7, 0.8], [0.4, 0.5])
    assert (c["threshold"], c["separable"]) == (0.6, True)
    c = tp.choose_threshold([0.604], [0.601])  # 中点 0.6025 取整为 0.60 会小于 max U，改保留 4 位
    assert (c["threshold"], c["separable"]) == (0.6025, True)
    c = tp.choose_threshold([0.63, 0.7], [0.62, 0.65])
    assert (c["threshold"], c["separable"]) == (0.63, False)
    c = tp.choose_threshold([0.579, 0.9], [0.6])
    assert (c["threshold"], c["separable"]) == (0.57, False)
    assert tp.describe([]) == {"n": 0, "min": None, "median": None, "max": None}
    assert tp.describe([0.3, 0.1, 0.2]) == {"n": 3, "min": 0.1, "median": 0.2, "max": 0.3}

    cases = [
        {"id": "a1", "kind": "answerable", "style": "natural", "expected_source": "11-发票政策"},
        {"id": "a2", "kind": "answerable", "style": "keyword", "expected_source": "11-发票政策"},
        {"id": "n1", "kind": "unanswerable", "group": "near", "style": "natural"},
        {"id": "n2", "kind": "unanswerable", "group": "near", "style": "keyword"},
        {"id": "o1", "kind": "unanswerable", "group": "off", "style": "natural"},
    ]
    s = tp.analyze(cases, [0.70, 0.66, 0.68, 0.50, 0.30])
    assert (s["threshold"], s["separable"]) == (0.66, False)
    at = s["at_threshold"]
    assert at["answerable_kept"] == {"count": 2, "n": 2, "rate": 1.0}
    assert at["unanswerable_near_filtered"] == {"count": 1, "n": 2, "rate": 0.5}
    assert at["unanswerable_off_filtered"] == {"count": 1, "n": 1, "rate": 1.0}
    assert at["unanswerable_keyword_filtered"]["count"] == 1
    assert s["distributions"]["unanswerable_near"]["max"] == 0.68
    assert s["distributions"]["answerable_keyword"]["n"] == 1

    tp.validate_cases(cases, {"11-发票政策"})
    for bad in [
        [{"id": "x", "kind": "answerable", "style": "natural", "expected_source": "不存在"}],
        [{"id": "x", "kind": "unanswerable", "style": "natural"}],  # 缺 group
        [{"id": "x", "kind": "other", "style": "natural"}],
        [cases[0], cases[0]],  # 重复 id
    ]:
        try:
            tp.validate_cases(bad, {"11-发票政策"})
        except ValueError:
            continue
        raise AssertionError(bad)
    print("✅ 阈值规则：可分取中点、不可分取 min A 向下取整；过滤率按业务相近 / 无关、按问法分列")


def test_eval_tag_naming_and_meta():
    """run_eval --tag：结果 / 报告文件名带标签（不覆盖 latest 报告），meta 记录 provider / 阈值，不含 key。"""
    import json
    import sys, os
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals"))
    import rag
    import run_eval

    class FakeAgent:
        async def ainvoke(self, state):
            return {"messages": [*state["messages"], AIMessage(content="好的")]}

    tmp = tempfile.mkdtemp()
    cases_path = os.path.join(tmp, "tag_cases.json")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump({"cases": [{"id": "t1", "category": "robustness", "description": "正常",
                              "messages": [{"role": "user", "content": "你好"}],
                              "expect": {"reply_contains": ["好的"]}}]}, f, ensure_ascii=False)
    old = (run_eval.RESULTS_DIR, run_eval.build_agent, sys.argv)
    run_eval.RESULTS_DIR = os.path.join(tmp, "results")
    run_eval.build_agent = lambda: FakeAgent()
    try:
        sys.argv = ["run_eval.py", "--cases", cases_path, "--tag", "zhipu-embedding-3_t0.25"]
        run_async(run_eval.main())
        sys.argv = ["run_eval.py", "--cases", cases_path, "--tag", "bad tag/.."]
        try:
            run_async(run_eval.main())
        except SystemExit as e:
            assert e.code == 1
        else:
            raise AssertionError("非法 tag 应当退出")
    finally:
        run_eval.RESULTS_DIR, run_eval.build_agent, sys.argv = old

    files = sorted(os.listdir(os.path.join(tmp, "results")))
    result_file = [x for x in files if x.endswith(".json")][0]
    assert result_file.startswith("tag_cases_zhipu-embedding-3_t0.25_"), files
    assert "tag_cases_latest_report_zhipu-embedding-3_t0.25.md" in files, files
    assert "tag_cases_latest_report.md" not in files
    data = json.load(open(os.path.join(tmp, "results", result_file), encoding="utf-8"))
    meta = data["meta"]
    assert meta["tag"] == "zhipu-embedding-3_t0.25"
    assert meta["embedding_provider"] == rag.EMBEDDING_PROVIDER
    assert meta["similarity_threshold"] == rag.SIMILARITY_THRESHOLD
    assert "embedding_index" in meta and "API_KEY" not in json.dumps(meta)
    print("✅ run_eval --tag：文件名带标签、不覆盖 latest 报告，meta 记录 embedding 与阈值")


if __name__ == "__main__":
    test_full_loop()
    test_history_window_and_order()
    test_history_to_messages()
    test_retrieval_constants_match_rag()
    test_parse_status_hit()
    test_parse_status_not_found()
    test_parse_status_kb_not_built()
    test_parse_status_error()
    test_parse_status_unknown()
    test_parse_real_rag_search_output()
    test_eval_extract_tool_records()
    test_eval_llm_error_does_not_abort_run()
    test_chat_endpoint_and_index()
    test_embedding_config_parsing()
    test_check_index_consistency()
    test_search_returns_error_on_index_mismatch()
    test_build_refuses_mismatched_index()
    test_build_and_search_with_real_chroma()
    test_log_index_status()
    test_embedding_retry_and_requests()
    test_threshold_probe_rule()
    test_eval_tag_naming_and_meta()
