"""
V7 自动化评测运行器：对 Agent 的工具选择、边界行为、多轮记忆、RAG 检索、
鲁棒性做批量断言，输出汇总报告。

用法（在项目根目录）：
    python evals/run_eval.py                # 跑全部用例
    python evals/run_eval.py boundary       # 只跑某一类（传类别名）
    python evals/run_eval.py --cases evals/holdout_cases.json   # 跑验证集（结果文件以 holdout_cases_ 开头）
    python evals/run_eval.py --cases evals/holdout_cases.json --ids holdout_002,holdout_004   # 只重跑指定用例

设计说明：
- 不走 HTTP、不起 uvicorn：直接 build_agent() 调用，多轮编排逻辑复用
  main.py 的 history_to_messages + db 的历史读写，和线上 /chat 同一条链路。
- 打真实 LLM（llm.py 已固定 temperature=0，天然低随机），不做任何 mock。
- 评测会真实写库：工单会真建（结束后报告新建的 ticket_id，便于清理），
  会话历史用 eval- 前缀的独立 session_id，跑完自动清理，不污染正常数据。
- 评测集与断言逻辑分离：加用例只改 cases.json。
"""
import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func, select  # noqa: E402

import db  # noqa: E402
from agent import build_agent  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from main import history_to_messages  # noqa: E402
from retrieval_trace import STATUSES, parse_retrieval  # noqa: E402

# 结果文件里每个工具返回原文最多保留的字符数（结构化的 retrievals 不截断）
OUTPUT_KEEP_CHARS = 500

CASES_FILE = os.path.join(os.path.dirname(__file__), "cases.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

CATEGORY_NAMES = {
    "tool_select": "工具选择",
    "boundary": "边界拒绝",
    "memory": "多轮记忆",
    "rag": "RAG 检索",
    "robustness": "鲁棒性",
}


def extract_trace(result):
    """从图运行结果里提取（工具名, 参数）序列和最终回复，与 main.py 的口径一致。"""
    messages = result["messages"]
    trace = []
    pending = {}
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                pending[call["id"]] = {"name": call["name"], "args": call["args"]}
        if isinstance(m, ToolMessage):
            info = pending.get(m.tool_call_id)
            if info:
                trace.append(info)
    reply = messages[-1].content if messages else ""
    return trace, reply


def extract_tool_records(result):
    """只用于留档，不参与判定：提取（工具返回原文截断版, 结构化检索轨迹）。

    工具返回按 OUTPUT_KEEP_CHARS 截断；retrievals 是对每次 search_knowledge
    调用的完整解析（status / degraded / hits），用来区分"相似度低于阈值的降级
    （not_found）"和"embedding / 网络异常导致的降级（error）"。
    """
    pending = {}
    outputs, retrievals = [], []
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                pending[call["id"]] = {"name": call["name"], "args": call["args"]}
        if isinstance(m, ToolMessage):
            info = pending.get(m.tool_call_id)
            if not info:
                continue
            content = str(m.content)
            outputs.append(
                {
                    "name": info["name"],
                    "output": content[:OUTPUT_KEEP_CHARS],
                    "output_chars": len(content),
                    "truncated": len(content) > OUTPUT_KEEP_CHARS,
                }
            )
            if info["name"] == "search_knowledge":
                retrievals.append(
                    parse_retrieval(str(info["args"].get("query", "")), content).model_dump()
                )
    return outputs, retrievals


def retrieval_status_counts(retrievals) -> dict:
    counts = {s: 0 for s in STATUSES}
    for r in retrievals:
        counts[r["status"]] += 1
    return counts


def evaluate(expect, tool_calls, reply) -> list[str]:
    """断言逻辑：工具断言精确，回复断言宽松（关键词包含）。返回失败原因列表。"""
    failures = []
    names = [c["name"] for c in tool_calls]

    tc = expect.get("tools_called")
    if tc is not None:
        if tc == []:
            if names:
                failures.append(f"期望不调用任何工具，实际调用了 {names}")
        else:
            missing = [t for t in tc if t not in names]
            if missing:
                failures.append(f"缺少期望的工具调用 {missing}（实际调用 {names}）")

    order = expect.get("tools_called_in_order")
    if order:
        it = iter(names)
        if not all(t in it for t in order):
            failures.append(f"工具调用顺序不符：期望 {order}，实际 {names}")

    for t in expect.get("tools_forbidden", []):
        if t in names:
            failures.append(f"出现了被禁止的调用：{t}（实际 {names}）")

    for spec in expect.get("tool_args", []):
        ok = any(
            c["name"] == spec["name"]
            and all(c["args"].get(k) == v for k, v in spec["args"].items())
            for c in tool_calls
        )
        if not ok:
            failures.append(f"没有参数匹配的调用：{spec}（实际 {tool_calls}）")

    for kw in expect.get("reply_contains", []):
        if kw not in reply:
            failures.append(f"回复缺少关键词 '{kw}'（回复开头：{reply[:100]!r}）")

    for kw in expect.get("reply_not_contains", []):
        if kw in reply:
            failures.append(f"回复出现了不该出现的内容 '{kw}'")

    return failures


async def run_case(agent, case: dict, run_id: str) -> dict:
    """跑一个用例：多轮消息在同一 session 内顺序发送，断言针对最后一轮。"""
    session_id = case.get("session_id") or f"eval-{run_id}-{case['id']}"
    turn_replies = []

    for msg in case["messages"]:
        history = await db.load_history(session_id)
        input_messages = history_to_messages(history)
        input_messages.append(HumanMessage(content=msg["content"]))
        result = await agent.ainvoke({"messages": input_messages})
        trace, reply = extract_trace(result)
        turn_replies.append({"tools": trace, "reply": reply, "result": result})
        await db.save_message(session_id, "user", msg["content"])
        await db.save_message(session_id, "assistant", reply)

    last = turn_replies[-1]
    failures = evaluate(case.get("expect", {}), last["tools"], last["reply"])
    # 留档字段（只记录最后一轮，与断言对象一致），不影响 passed
    tool_outputs, retrievals = extract_tool_records(last["result"])
    return {
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "passed": not failures,
        "failures": failures,
        "tools_called": last["tools"],
        "reply": last["reply"],
        "session_id": session_id,
        "tool_outputs": tool_outputs,
        "retrievals": retrievals,
        "retrieval_status_counts": retrieval_status_counts(retrievals),
    }


async def main():
    parser = argparse.ArgumentParser(description="Agent 自动化评测")
    parser.add_argument("category", nargs="?", help="只跑某一类用例（类别名）")
    parser.add_argument(
        "--cases",
        default=CASES_FILE,
        help="用例文件路径，默认 evals/cases.json；例如验证集：--cases evals/holdout_cases.json",
    )
    parser.add_argument(
        "--ids",
        help="只跑指定用例，逗号分隔，例如 --ids holdout_002,holdout_004；"
        "结果与报告会标注为重跑，不覆盖完整运行的 latest 报告",
    )
    args = parser.parse_args()
    category_filter = args.category
    id_filter = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None

    cases_path = args.cases
    if not os.path.exists(cases_path):  # 允许只写 evals/ 下的文件名
        cases_path = os.path.join(os.path.dirname(__file__), args.cases)
    cases_name = os.path.basename(cases_path)
    # 默认用例集沿用原来的文件名；其他用例集加前缀，避免覆盖 / 混淆
    is_default = os.path.abspath(cases_path) == os.path.abspath(CASES_FILE)
    cases_stem = os.path.splitext(cases_name)[0]

    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter]
        if not cases:
            print(f"没有类别为 {category_filter} 的用例，可选：{list(CATEGORY_NAMES)}")
            sys.exit(1)

    if id_filter:
        known = {c["id"] for c in cases}
        unknown = [i for i in id_filter if i not in known]
        if unknown:
            print(f"用例集中没有这些 id：{unknown}")
            sys.exit(1)
        cases = [c for c in cases if c["id"] in id_filter]

    await db.ensure_tables()

    # 记录运行前最大工单号，结束后报告新建的工单（评测会真实写库）
    async with db.AsyncSessionLocal() as session:
        baseline_ticket = await session.scalar(select(func.max(db.Ticket.ticket_id))) or 0
    print(f"开始评测：{len(cases)} 个用例，真实 LLM（会真实建工单，结束后报告新建工单）")
    print("=" * 60)

    agent = build_agent()
    run_id = uuid.uuid4().hex[:8]
    started = datetime.now()
    records = []
    for case in cases:
        record = await run_case(agent, case, run_id)
        records.append(record)
        mark = "✅" if record["passed"] else "❌"
        print(f"{mark} [{record['category']}] {record['id']}: {record['description']}")
        for f in record["failures"]:
            print(f"     ↳ {f}")

    # ---- 汇总报告 ----
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    by_cat: dict[str, list] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    stamp = started.strftime("%Y%m%d_%H%M%S")
    duration = (datetime.now() - started).total_seconds()

    lines = [
        "# Agent 评测报告",
        "",
        f"- 运行时间：{started.strftime('%Y-%m-%d %H:%M:%S')}（耗时 {duration:.0f}s）",
        f"- 用例集：{cases_name}",
        f"- 模型：{os.getenv('LLM_MODEL_NAME', '?')}（temperature=0）",
        f"- 总用例：{total}，通过：{passed}，**通过率 {passed / total * 100:.0f}%**",
    ]
    if id_filter:
        lines.append(f"- ⚠️ 本次是指定用例的重跑（--ids {','.join(id_filter)}），不是完整运行")
    # 检索状态汇总（只记录，不参与判定）：error = embedding / 网络等异常，not_found = 全部低于相似度阈值
    status_total = {s: 0 for s in STATUSES}
    for r in records:
        for s, n in r["retrieval_status_counts"].items():
            status_total[s] += n
    lines.append(
        "- 本次检索状态（search_knowledge 调用次数）："
        + "，".join(f"{s} {n}" for s, n in status_total.items())
    )
    error_ids = [r["id"] for r in records if r["retrieval_status_counts"]["error"] > 0]
    if error_ids:
        lines.append(f"- ⚠️ 检索异常（status=error，可能是基础设施故障而非模型问题）的用例：{'、'.join(error_ids)}")
    unknown_ids = [r["id"] for r in records if r["retrieval_status_counts"]["unknown"] > 0]
    if unknown_ids:
        lines.append(f"- ⚠️ 检索返回格式无法识别（status=unknown）的用例：{'、'.join(unknown_ids)}")
    lines += [
        "",
        "| 类别 | 用例数 | 通过 | 通过率 |",
        "| --- | --- | --- | --- |",
    ]
    for cat, rs in by_cat.items():
        p = sum(1 for r in rs if r["passed"])
        lines.append(
            f"| {CATEGORY_NAMES.get(cat, cat)} | {len(rs)} | {p} | {p / len(rs) * 100:.0f}% |"
        )

    failed = [r for r in records if not r["passed"]]
    if failed:
        lines += ["", "## 失败用例详情", ""]
        for r in failed:
            lines.append(f"### {r['id']}（{CATEGORY_NAMES.get(r['category'])}）")
            lines.append(f"- {r['description']}")
            for f in r["failures"]:
                lines.append(f"- ❌ {f}")
            lines.append(f"- 实际工具调用：`{json.dumps(r['tools_called'], ensure_ascii=False)}`")
            lines.append(f"- 实际回复开头：{r['reply'][:150]!r}")
            lines.append(f"- 检索状态：{json.dumps(r['retrieval_status_counts'], ensure_ascii=False)}")
            lines.append("")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_prefix = "eval" if is_default else cases_stem
    # 重跑（--ids）的结果文件加 rerun 标记，报告不覆盖完整运行的 latest 报告
    rerun_tag = "_rerun" if id_filter else ""
    result_file = os.path.join(RESULTS_DIR, f"{result_prefix}{rerun_tag}_{stamp}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": {"time": stamp, "model": os.getenv("LLM_MODEL_NAME"), "total": total, "passed": passed,
                      "cases_file": cases_name, "ids": id_filter,
                      "retrieval_error_ids": error_ids, "retrieval_unknown_ids": unknown_ids},
             "results": records},
            f, ensure_ascii=False, indent=2,
        )
    report_stem = "latest_report" if is_default else f"{cases_stem}_latest_report"
    report_name = f"{report_stem}{'_rerun' if id_filter else ''}.md"
    report_file = os.path.join(RESULTS_DIR, report_name)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 60)
    print(f"汇总：{passed}/{total} 通过（{passed / total * 100:.0f}%）")
    for cat, rs in by_cat.items():
        p = sum(1 for r in rs if r["passed"])
        print(f"  {CATEGORY_NAMES.get(cat, cat):6s} {p}/{len(rs)}（{p / len(rs) * 100:.0f}%）")
    print(f"结果已保存：{result_file}")
    print(f"报告已生成：{report_file}")

    # ---- 写库审计：新建工单报告 + 评测会话清理 ----
    async with db.AsyncSessionLocal() as session:
        new_tickets = (
            (await session.execute(
                select(db.Ticket).where(db.Ticket.ticket_id > baseline_ticket)
            ))
            .scalars().all()
        )
    if new_tickets:
        print(f"\n⚠️ 本次评测新建了 {len(new_tickets)} 个工单：")
        for t in new_tickets:
            print(f"  ticket_id={t.ticket_id} order={t.order_id} reason={t.reason[:40]}")
        print("  如需清理：DELETE FROM tickets WHERE ticket_id > %d;" % baseline_ticket)

    session_ids = {r["session_id"] for r in records}
    deleted = 0
    for s in session_ids:
        deleted += await db.clear_session(s)
    print(f"已清理评测会话历史 {deleted} 条（{len(session_ids)} 个 session）。")


if __name__ == "__main__":
    asyncio.run(main())
