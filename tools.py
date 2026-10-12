"""
三个工具：query_order / search_knowledge / create_ticket。

V5 起状态：三个工具全部接真数据 —— search_knowledge 走 RAG 向量检索（rag.py），
query_order / create_ticket 走 PostgreSQL（db.py，SQLAlchemy 异步 + asyncpg）。

异步说明：LangGraph 的 ToolNode 有两条执行路径——同步图 .invoke 时把同步工具
丢进线程池，异步图 .ainvoke 时直接 await 异步工具。V5 起主链路走 ainvoke
（见 main.py），所以 DB 工具是 async 的，asyncpg 连接全程在同一个事件循环上，
DB 阻塞不会卡住 FastAPI。search_knowledge 保持同步（rag.py 是同步实现），
ToolNode 在异步路径下会自动把它放进线程池执行，同样不阻塞事件循环。
"""
import json

from langchain_core.tools import tool

import db
import rag


def _dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


@tool
async def query_order(order_id: str) -> str:
    """根据订单号查询订单状态和物流信息。输入订单号字符串，返回订单详情。"""
    try:
        await db.ensure_tables()
        async with db.session_scope() as session:
            order = await session.get(db.Order, order_id)
        if order is None:
            # 查不到订单是正常业务分支，返回结构化错误让模型组织话术，不抛异常
            return _dumps({"error": f"订单 {order_id} 不存在，请确认订单号是否正确"})
        return _dumps(
            {
                "order_id": order.order_id,
                "user_id": order.user_id,
                "status": order.status,
                "logistics_status": order.logistics_status,
                "logistics_detail": order.logistics_detail,
                "created_at": order.created_at,
            }
        )
    except Exception as e:  # 连不上库 / 建表失败等：服务不崩，给模型一个明确错误
        return _dumps({"error": f"数据库暂时不可用：{e}"})


@tool
def search_knowledge(query: str) -> str:
    """在企业知识库中检索与用户问题相关的处理规则或FAQ。输入用户问题，返回相关知识片段及其来源。"""
    # V3：真实 RAG。函数签名与 V0 完全一致（输入query，输出相关片段字符串），
    # 内部从"关键词假匹配"换成了 Embedding + ChromaDB 向量检索（见 rag.py），
    # 上层 Agent 图结构零改动——这就是当初面向接口设计换来的好处。
    #
    # 关于异步安全：rag.search 是同步阻塞的（ChromaDB 查询 + embedding HTTP 请求），
    # ToolNode 在 ainvoke 路径下会把这个同步工具放进线程池执行，
    # 阻塞发生在工作线程里，不会卡住 FastAPI 的事件循环。
    return rag.search(query)


@tool
async def create_ticket(order_id: str, reason: str) -> str:
    """当判断需要人工介入时创建工单。输入订单号和创建原因，返回生成的工单号。"""
    try:
        await db.ensure_tables()
        async with db.session_scope() as session:
            # 先确认订单存在，给模型一个可读的错误，而不是等外键约束炸出 IntegrityError
            order = await session.get(db.Order, order_id)
            if order is None:
                return _dumps({"error": f"订单 {order_id} 不存在，无法创建工单"})
            ticket = db.Ticket(order_id=order_id, reason=reason, status="待处理")
            session.add(ticket)
            await session.flush()  # 拿到自增主键 ticket_id
            result = {
                "ticket_id": ticket.ticket_id,
                "order_id": ticket.order_id,
                "reason": ticket.reason,
                "status": ticket.status,
            }
        return _dumps(result)
    except Exception as e:
        return _dumps({"error": f"数据库暂时不可用，工单未创建：{e}"})


ALL_TOOLS = [query_order, search_knowledge, create_ticket]
