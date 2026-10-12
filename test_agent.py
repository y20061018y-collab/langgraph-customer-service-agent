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
    test_eval_extract_tool_records()
    test_chat_endpoint_and_index()
