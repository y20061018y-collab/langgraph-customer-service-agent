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


def test_parse_retrieval():
    """main.parse_retrieval 能还原 rag.search 的命中格式与三种降级文案。"""
    import rag
    from main import parse_retrieval

    # 与 rag.search 命中时的输出格式一致（含多行片段与结尾提示）
    hit_output = (
        "从知识库中检索到以下相关片段：\n\n"
        "[1]（来源: 11-发票政策，相似度 0.62）\n发票政策\n支持电子普通发票和增值税专用发票。\n\n"
        "[2]（来源: 03-退款流程与时效，相似度 0.31）\n退款流程\n仅退款适用于未收到货。"
        "\n\n（如果以上片段与用户问题无关，请直接告知用户知识库中没有相关内容，不要编造。）"
    )
    r = parse_retrieval("怎么开发票", hit_output)
    assert (r.status, r.degraded, r.query) == ("hit", False, "怎么开发票")
    assert [h.source for h in r.hits] == ["11-发票政策", "03-退款流程与时效"]
    assert [h.similarity for h in r.hits] == [0.62, 0.31]
    assert r.hits[0].snippet == "发票政策\n支持电子普通发票和增值税专用发票。"
    assert "如果以上片段" not in r.hits[1].snippet

    for output, status in [
        (rag.NOT_FOUND_MSG, "not_found"),
        (rag.KB_NOT_BUILT_MSG, "kb_not_built"),
        ("知识库检索失败：boom。请告知用户暂时无法查询知识库。", "error"),
    ]:
        r = parse_retrieval("q", output)
        assert (r.status, r.degraded, r.hits) == (status, True, [])
    print("✅ 检索轨迹解析正确：命中（含来源/相似度）与三种降级")


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
    test_parse_retrieval()
    test_chat_endpoint_and_index()
