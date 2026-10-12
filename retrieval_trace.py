"""
把 rag.search 返回的文本还原成结构化的检索轨迹。

这是一个无副作用的小模块：只依赖标准库和 pydantic，不 import rag / db / agent，
所以 main.py（HTTP 接口）和 evals/run_eval.py（评测）都可以直接导入它，
不会触发 chromadb 加载、读 .env 或连数据库。

解析依赖 rag.search 的输出格式（片段头部和三条降级文案）。降级文案在这里
重复了一份字符串常量（不能 import rag）；test_agent.py 里有测试断言它们与
rag.py 中的常量一致，rag.py 改了文案而这里没同步，测试会失败。

status 取值：
- hit           检索到至少一个片段（相似度已过阈值）
- not_found     检索正常，但全部结果低于相似度阈值
- kb_not_built  知识库尚未建立
- error         检索过程抛异常（例如 embedding 接口报错 / 网络故障）
- unknown       输出格式无法识别——不猜测，也不当作 hit
"""
import re

from pydantic import BaseModel

# 与 rag.NOT_FOUND_MSG / rag.KB_NOT_BUILT_MSG 保持一致（见模块说明）
NOT_FOUND_MSG = "未找到相关信息"
KB_NOT_BUILT_MSG = "知识库尚未建立，请先运行 python scripts/build_kb.py 建库"
ERROR_PREFIX = "知识库检索失败"

STATUSES = ("hit", "not_found", "kb_not_built", "error", "unknown")


class RetrievalHit(BaseModel):
    source: str
    similarity: float
    snippet: str


class RetrievalTrace(BaseModel):
    """一次 search_knowledge 调用的结构化结果。

    degraded：not_found / kb_not_built / error 为 True，hit 为 False；
    unknown 时为 None（无法判断，不下结论）。
    """

    query: str
    status: str
    degraded: bool | None
    hits: list[RetrievalHit] = []


# rag.search 命中时每个片段的头部格式：[1]（来源: 文件名，相似度 0.52）
_HIT_HEADER = re.compile(r"^\[(\d+)\]（来源: (.*?)，相似度 (-?[\d.]+)）\n", re.MULTILINE)
_HIT_FOOTER = "\n\n（如果以上片段"


def parse_retrieval(query: str, output: str) -> RetrievalTrace:
    if output == KB_NOT_BUILT_MSG:
        return RetrievalTrace(query=query, status="kb_not_built", degraded=True)
    if output == NOT_FOUND_MSG:
        return RetrievalTrace(query=query, status="not_found", degraded=True)
    if output.startswith(ERROR_PREFIX):
        return RetrievalTrace(query=query, status="error", degraded=True)

    headers = list(_HIT_HEADER.finditer(output))
    hits = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(output)
        body = output[m.end():end]
        footer = body.find(_HIT_FOOTER)
        if footer != -1:
            body = body[:footer]
        try:
            similarity = float(m.group(3))
        except ValueError:  # 形如 "1.2.3" 的异常分数：整体按无法识别处理
            return RetrievalTrace(query=query, status="unknown", degraded=None)
        hits.append(RetrievalHit(source=m.group(2), similarity=similarity, snippet=body.strip()))
    if not hits:
        return RetrievalTrace(query=query, status="unknown", degraded=None)
    return RetrievalTrace(query=query, status="hit", degraded=False, hits=hits)
