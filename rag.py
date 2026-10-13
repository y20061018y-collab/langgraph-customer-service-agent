"""
RAG 模块（V3）：知识库的加载、切分、向量化入库与检索。

整体流水线：
    knowledge/*.md --加载--> 原始文档 --切分--> chunks
        --embedding-3向量化--> ChromaDB(持久化 ./chroma_db) --检索--> top_k 片段

设计决定（README 的 V3 章节有完整版）：
- 不用 LangChain 的 DocumentLoader / TextSplitter / VectorStore 封装，
  只用 chromadb 原生 API —— 每个环节自己写，每个参数都要能讲清楚为什么。
- Embedding 走 OpenAI 兼容的 POST /embeddings 接口（智谱 embedding-3），
  base_url / key 从 .env 读取，与 LLM 共用同一套配置机制，换厂商只改配置。
- build_knowledge_base() 幂等：集合里已有数据就直接跳过，可以反复执行。
- search() 永不抛异常：知识库为空、检索无结果、embedding 接口报错，
  都降级为一段明确的文字返回给 Agent，由 Agent 决定怎么向用户解释。

线程模型说明：ChromaDB 查询（SQLite）和 embedding HTTP 请求都是同步阻塞的，
但本模块整体保持"纯同步函数"—— LangGraph 的 ToolNode 对同步工具是通过
线程池执行器（get_executor_for_config）来跑的，所以阻塞发生在工作线程里，
不会卡住 FastAPI 的事件循环（详见 tools.py 里 search_knowledge 的注释）。
"""
import os

import chromadb
import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------- 路径与配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_base"

# embedding 模型与接口配置：优先用 EMBEDDING_* 专属配置，
# 没配就复用 LLM_* 的 base_url / key（两者都是 OpenAI 兼容协议）。
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "embedding-3")
EMBEDDING_BASE_URL = (os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL") or "").rstrip("/")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
# 智谱 embedding-3 支持的输出维度：256 / 512 / 1024 / 2048，默认 1024。
# 维度越小存储和计算越省，1024 是召回精度和成本的平衡点。
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# 切分参数：chunk 太大则一段里混了多个主题，向量语义被稀释；
# 太小则上下文不完整。中文政策类文本一屏在 200-400 字，350 是经验值。
CHUNK_SIZE = 350
# 相邻 chunk 重叠的字符数：防止关键句子正好被切在边界上一分为二，
# 导致任何单个 chunk 都读不懂完整语义。
CHUNK_OVERLAP = 60

# 相似度阈值：低于它的检索结果直接判为"未找到"。
# 向量检索的 top_k 永远会返回结果（哪怕毫不相关），不加阈值的话
# 乱问的问题也会召回一条最高分的垃圾片段，Agent 容易拿去编答案。
# 0.25 是保守值，可按真实 embedding 的分数分布用环境变量调整；设为 0 可关闭。
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.25"))

NOT_FOUND_MSG = "未找到相关信息"
KB_NOT_BUILT_MSG = "知识库尚未建立，请先运行 python scripts/build_kb.py 建库"

# 进程内缓存 collection 句柄，避免每次检索都重新打开 SQLite。
_collection_cache = None


# ---------- 1. 文档加载 ----------
def load_documents() -> list[dict]:
    """扫描 knowledge/ 目录下所有 .md / .txt 文件，返回 [{source, text}]。

    source 用文件名（不含扩展名），它就是最后拼进答案里的"来源标注"。
    """
    docs = []
    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(KNOWLEDGE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append({"source": os.path.splitext(filename)[0], "text": text})
    return docs


# ---------- 2. 切分（chunking）----------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把一篇文档切成若干个 chunk。

    策略分两层：
    1. 先按空行 / markdown 标题切成"自然段"，语义边界天然在这里；
    2. 贪心地合并自然段：不超过 chunk_size 就继续装，装不下就开新 chunk；
    3. 如果单个自然段本身就超过 chunk_size（罕见），用滑动窗口硬切，
       窗口步长 = chunk_size - overlap，保证相邻窗口有 overlap 的重叠。

    这是"结构感知 + 长度兜底"的朴素做法。对 300 字左右的政策类短文档，
    大多数文档只会产出 1 个 chunk，这是符合预期的：一篇短文本来就是一个语义单元。
    """
    # 把标题行和空行都当作段落分隔符
    paragraphs: list[str] = []
    for block in text.replace("\r\n", "\n").split("\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):  # markdown 标题行单独成段，方便归入下文
            if paragraphs and paragraphs[-1] != "":
                paragraphs.append("")
            paragraphs.append(block.lstrip("#").strip())
            paragraphs.append("")
        else:
            paragraphs.append(block)

    # 重新按空行聚合出自然段
    natural: list[str] = []
    buf: list[str] = []
    for p in paragraphs:
        if p == "":
            if buf:
                natural.append("\n".join(buf))
                buf = []
        else:
            buf.append(p)
    if buf:
        natural.append("\n".join(buf))

    # 贪心合并自然段
    chunks: list[str] = []
    current = ""
    for para in natural:
        if not current:
            current = para
        elif len(current) + len(para) + 1 <= chunk_size:
            current = current + "\n" + para
        else:
            chunks.append(current)
            current = para
        # 单段超长的兜底：滑动窗口硬切
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)
    return chunks


# ---------- 3. 向量化（embedding）----------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """调用 OpenAI 兼容的 /embeddings 接口批量向量化。

    用 httpx 直接发 HTTP 请求而不是任何 SDK 封装，是因为这个接口本身就是
    一个简单的 POST：请求体 {model, input, dimensions}，这一层自己写，
    出问题时（换厂商、报错排查）每一行都是自己看得懂的。
    """
    if not EMBEDDING_BASE_URL or not EMBEDDING_API_KEY:
        raise RuntimeError(
            "缺少 embedding 配置：请在 .env 中配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY"
            "（智谱 embedding-3 需要智谱开放平台的 key）"
        )
    url = f"{EMBEDDING_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}

    all_vectors: list[list[float] | None] = [None] * len(texts)
    # 分批调用：embedding 接口对单次输入条数有限制，16 条一批是稳妥值
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = httpx.post(
            url,
            headers=headers,
            json={
                "model": EMBEDDING_MODEL_NAME,
                "input": batch,
                "dimensions": EMBEDDING_DIMENSIONS,
            },
            timeout=60.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"embedding 接口返回 {resp.status_code}（model={EMBEDDING_MODEL_NAME}，"
                f"base_url={EMBEDDING_BASE_URL}）：{resp.text[:300]}"
            )
        data = resp.json()["data"]  # 返回值带 index 字段，按它排回原顺序
        for item in data:
            all_vectors[start + item["index"]] = item["embedding"]
    return all_vectors  # type: ignore[return-value]


# ---------- 4. ChromaDB 集合 ----------
def get_collection():
    """获取（或创建）持久化的 ChromaDB 集合。

    hnsw:space 用 cosine 而不是默认的 l2：cosine 只关心向量方向（语义相似度），
    不关心模长，对文本 embedding 是标准选择。
    """
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache


# ---------- 5. 建库（幂等）----------
def build_knowledge_base(rebuild: bool = False) -> dict:
    """加载知识文档 → 切分 → 向量化 → 写入 ChromaDB。

    幂等逻辑：集合里已经有数据且未指定 rebuild 时直接跳过，
    所以重复运行、或者服务启动时误调用它，都不会重复入库。

    返回统计信息 {files, chunks, skipped}。
    """
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"knowledge/ 目录下没有找到任何 .md/.txt 文档：{KNOWLEDGE_DIR}")

    collection = get_collection()
    if collection.count() > 0 and not rebuild:
        print(f"知识库已存在（{collection.count()} 个 chunk），跳过建库。如需重建：python scripts/build_kb.py --rebuild")
        return {"files": len(docs), "chunks": collection.count(), "skipped": True}

    if rebuild:
        client = collection._client  # noqa: SLF001 重建时直接删集合最干净
        client.delete_collection(COLLECTION_NAME)
        global _collection_cache
        _collection_cache = None
        collection = get_collection()

    # 所有 chunk 平铺，并记录来源文档，作为 metadata 存进去
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})

    print(f"加载 {len(docs)} 篇文档，切分出 {len(texts)} 个 chunk，开始向量化...")
    vectors = embed_texts(texts)

    # 批量写入。id 用 文件名::序号，既唯一又便于人工排查。
    collection.add(
        ids=[f"{m['source']}::{i}" for i, m in enumerate(metadatas)],
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"建库完成：{collection.count()} 个 chunk 已写入 {CHROMA_DIR}")
    return {"files": len(docs), "chunks": len(texts), "skipped": False}


# ---------- 6. 检索 ----------
def search(query: str, top_k: int = 3) -> str:
    """向量检索：query 向量化 → ChromaDB 余弦相似度 → top_k 片段。

    返回给上层的是一段带来源标注的纯文本上下文，Agent 拿到它后
    组织成自然语言回复。注意这个函数永远不抛异常：
    - 知识库没建 / 为空   → 提示先建库
    - 检索结果为空        → "未找到相关信息"
    - embedding 接口挂了  → 明确的失败说明
    这样 Agent 在任何情况下都能把结果继续往下说，而不是整条链路 500。
    """
    try:
        collection = get_collection()
        if collection.count() == 0:
            return KB_NOT_BUILT_MSG

        query_vector = embed_texts([query])[0]
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"][0]
        if not documents:
            return NOT_FOUND_MSG

        # cosine distance ∈ [0, 2]，相似度 = 1 - distance，越大越相关
        hits = []
        for doc, meta, dist in zip(documents, result["metadatas"][0], result["distances"][0]):
            similarity = 1 - dist
            if similarity < SIMILARITY_THRESHOLD:
                continue  # 低于阈值视为不相关，过滤掉而不是喂给模型
            hits.append((similarity, meta.get("source", "未知来源"), doc))
        if not hits:
            return NOT_FOUND_MSG
        lines = [
            f"[{i}]（来源: {source}，相似度 {similarity:.2f}）\n{doc}"
            for i, (similarity, source, doc) in enumerate(hits, start=1)
        ]
        return (
            "从知识库中检索到以下相关片段：\n\n"
            + "\n\n".join(lines)
            + "\n\n（如果以上片段与用户问题无关，请直接告知用户知识库中没有相关内容，不要编造。）"
        )
    except Exception as e:  # noqa: BLE001 检索失败不能拖垮整个 Agent 链路
        return f"知识库检索失败：{e}。请告知用户暂时无法查询知识库。"
