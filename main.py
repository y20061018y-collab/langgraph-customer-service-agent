"""
核心接口是 POST /chat

返回结构里特意把"这一轮到底调用了哪些工具、传了什么参数、拿到了什么结果"
也吐出来 —— 这条轨迹不是给用户看的，是给你自己调试、以及以后面试时
"打开开发者工具给面试官看执行过程"用的。

V6 起支持多轮会话：请求可携带 session_id，服务端从 PostgreSQL 的 messages
表恢复最近 MAX_HISTORY_MESSAGES 条历史，以多轮 messages 数组形式注入 Agent；
本轮结束后把用户消息和最终回复写回库里（工具调用中间过程不入库）。
不传 session_id 时自动生成并随响应返回，老用法完全兼容。
"""
import os
import re
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel

import db
import rag
from agent import build_agent

app = FastAPI(title="企业客服与工单 Agent")

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ToolCallTrace(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str


class RetrievalHit(BaseModel):
    source: str
    similarity: float
    snippet: str


class RetrievalTrace(BaseModel):
    """一次 search_knowledge 调用的结构化结果（由工具输出文本解析而来）。

    status: hit（有命中）/ not_found（全部低于阈值）/ kb_not_built（知识库未建）/
    error（检索异常）；后三种都属于降级，degraded=True。
    """

    query: str
    status: str
    degraded: bool
    hits: list[RetrievalHit] = []


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCallTrace]
    session_id: str
    # 向后兼容的新增字段：旧客户端忽略即可
    retrievals: list[RetrievalTrace] = []


# rag.search 命中时每个片段的头部格式：[1]（来源: 文件名，相似度 0.52）
_HIT_HEADER = re.compile(r"^\[(\d+)\]（来源: (.*?)，相似度 (-?[\d.]+)）\n", re.MULTILINE)
_HIT_FOOTER = "\n\n（如果以上片段"


def parse_retrieval(query: str, output: str) -> RetrievalTrace:
    """把 rag.search 返回的文本还原成结构化的检索轨迹。

    只读不改：agent / rag 的行为不受影响。解析依赖 rag.search 的输出格式
    （片段头部与三条降级文案），rag.py 改了输出格式这里要同步改。
    """
    if output == rag.KB_NOT_BUILT_MSG:
        return RetrievalTrace(query=query, status="kb_not_built", degraded=True)
    if output == rag.NOT_FOUND_MSG:
        return RetrievalTrace(query=query, status="not_found", degraded=True)
    if output.startswith("知识库检索失败"):
        return RetrievalTrace(query=query, status="error", degraded=True)

    headers = list(_HIT_HEADER.finditer(output))
    hits = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(output)
        body = output[m.end():end]
        footer = body.find(_HIT_FOOTER)
        if footer != -1:
            body = body[:footer]
        hits.append(
            RetrievalHit(source=m.group(2), similarity=float(m.group(3)), snippet=body.strip())
        )
    if not hits:  # 格式无法识别：不猜测，按降级报告
        return RetrievalTrace(query=query, status="error", degraded=True)
    return RetrievalTrace(query=query, status="hit", degraded=False, hits=hits)


def history_to_messages(rows) -> list[AnyMessage]:
    """把 messages 表的行转成 LangChain 消息对象（多轮 messages 数组）。

    以结构化多轮消息传给模型，而不是拼成一段"以下是历史对话"的文本：
    模型对 messages 数组里角色边界的理解远好于自由文本，指代消解（"它"）
    也更稳。
    """
    messages: list[AnyMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # V5：改用 ainvoke 异步执行。工具已是 async（真查/真写 PostgreSQL），
    # 异步路径下 asyncpg 全程跑在 FastAPI 的事件循环上，DB 等待不阻塞服务。
    agent = get_agent()

    session_id = req.session_id or str(uuid.uuid4())

    # 恢复上下文：滑动窗口内的历史 + 本轮用户消息，一起进图。
    # 图的 agent 节点会在最前面补 SystemMessage（见 agent.py），工具循环不变。
    history_rows = await db.load_history(session_id)
    input_messages = history_to_messages(history_rows)
    input_messages.append(HumanMessage(content=req.message))
    result = await agent.ainvoke({"messages": input_messages})

    messages = result["messages"]

    # 把这一轮里所有的 tool_calls 和对应的 ToolMessage 结果配对，组装成轨迹
    trace: list[ToolCallTrace] = []
    retrievals: list[RetrievalTrace] = []
    pending_calls = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                pending_calls[call["id"]] = {"name": call["name"], "args": call["args"]}
        if isinstance(m, ToolMessage):
            call_info = pending_calls.get(m.tool_call_id, {"name": "unknown", "args": {}})
            trace.append(
                ToolCallTrace(
                    tool_name=call_info["name"],
                    tool_input=call_info["args"],
                    tool_output=m.content,
                )
            )
            if call_info["name"] == "search_knowledge":
                retrievals.append(
                    parse_retrieval(str(call_info["args"].get("query", "")), str(m.content))
                )

    final_reply = messages[-1].content if messages else ""

    # 本轮对话入库：只存 user 消息 + assistant 最终回复（中间工具过程不入库）
    await db.save_message(session_id, "user", req.message)
    await db.save_message(session_id, "assistant", final_reply)

    return ChatResponse(
        reply=final_reply, tool_calls=trace, session_id=session_id, retrievals=retrievals
    )


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    rows = await db.load_history(session_id, limit=1000)
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at,
            }
            for r in rows
        ],
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    deleted = await db.clear_session(session_id)
    return {"session_id": session_id, "deleted": deleted}


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


@app.get("/", include_in_schema=False)
def index():
    """演示界面（静态单页，见 web/index.html）。"""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
