"""
RAG 模块（V3）：知识库的加载、切分、向量化入库与检索。

整体流水线：
    knowledge/*.md --加载--> 原始文档 --切分--> chunks
        --embedding 向量化（智谱 embedding-3 / Ollama bge-m3）--> ChromaDB(持久化 ./chroma_db)
        --检索--> top_k 片段

设计决定（README 的 V3 章节有完整版）：
- 不用 LangChain 的 DocumentLoader / TextSplitter / VectorStore 封装，
  只用 chromadb 原生 API —— 每个环节自己写，每个参数都要能讲清楚为什么。
- Embedding 默认走 OpenAI 兼容的 POST /embeddings 接口（智谱 embedding-3），
  base_url / key 从 .env 读取，与 LLM 共用同一套配置机制；
  EMBEDDING_PROVIDER=ollama 时改走本地 Ollama 的 POST /api/embed。
- 集合元数据记录建库时的 provider / model / 维度；与当前配置不一致时
  明确失败（search 返回"知识库检索失败：…"，建库要求 --rebuild），不会悄悄用错的向量检索。
- embedding 请求对连接类错误最多重试 3 次（间隔 1s、3s），每次失败都写日志。
- build_knowledge_base() 幂等：集合里已有数据就直接跳过，可以反复执行。
- search() 永不抛异常：知识库为空、检索无结果、embedding 接口报错，
  都降级为一段明确的文字返回给 Agent，由 Agent 决定怎么向用户解释。

线程模型说明：ChromaDB 查询（SQLite）和 embedding HTTP 请求都是同步阻塞的，
但本模块整体保持"纯同步函数"—— LangGraph 的 ToolNode 对同步工具是通过
线程池执行器（get_executor_for_config）来跑的，所以阻塞发生在工作线程里，
不会卡住 FastAPI 的事件循环（详见 tools.py 里 search_knowledge 的注释）。
"""
import logging
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import chromadb
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag")

# ---------- 路径与配置 ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_base"

# embedding 提供方：
# - zhipu：OpenAI 兼容的 POST {base}/embeddings（智谱 embedding-3），默认值，行为与引入 provider 之前一致
# - ollama：本地 Ollama 的 POST {base}/api/embed（如 bge-m3），不需要 key
PROVIDERS = ("zhipu", "ollama")
DEFAULT_MODELS = {"zhipu": "embedding-3", "ollama": "bge-m3"}
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
# 没有元数据的旧库（引入一致性校验之前建的）只在这组默认配置下放行，见 check_index
LEGACY_DEFAULT = ("zhipu", "embedding-3", 1024)


class EmbeddingConfigError(RuntimeError):
    """环境变量取值非法（provider 拼错、数字解析失败等）。"""


class IndexMismatchError(RuntimeError):
    """向量库建库时的 provider / model / 维度与当前配置不一致。"""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None = field(repr=False)  # repr 不带 key，避免被日志或报错带出去
    # 请求维度：只有 zhipu 会放进请求体；ollama 的维度由模型决定，以实际返回为准
    dimensions: int
    similarity_threshold: float
    timeout: float

    def describe(self) -> str:
        dim = self.dimensions if self.provider == "zhipu" else "?"
        return f"{self.provider}/{self.model}/{dim}"


def load_embedding_config(env) -> EmbeddingConfig:
    """从环境变量（任意 Mapping）解析 embedding 配置。纯函数，便于单测。

    缺 key / base_url 不在这里报错（与引入 provider 之前一致，留到真正调用接口时报），
    这里只拦截"取值本身非法"的情况。
    """
    provider = (env.get("EMBEDDING_PROVIDER") or "zhipu").strip().lower()
    if provider not in PROVIDERS:
        raise EmbeddingConfigError(f"EMBEDDING_PROVIDER 只能是 {'/'.join(PROVIDERS)}，当前为 {provider!r}")
    model = env.get("EMBEDDING_MODEL_NAME") or DEFAULT_MODELS[provider]
    if provider == "zhipu":
        # 优先用 EMBEDDING_* 专属配置，没配就复用 LLM_* 的 base_url / key（两者都是 OpenAI 兼容协议）
        base_url = env.get("EMBEDDING_BASE_URL") or env.get("LLM_BASE_URL") or ""
        api_key = env.get("EMBEDDING_API_KEY") or env.get("LLM_API_KEY")
    else:
        # Ollama 不回退到 LLM_BASE_URL：那是对话模型的地址，不是 Ollama
        base_url = env.get("EMBEDDING_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
        api_key = None
    try:
        # 智谱 embedding-3 支持的输出维度：256 / 512 / 1024 / 2048，默认 1024。
        # 维度越小存储和计算越省，1024 是召回精度和成本的平衡点。
        dimensions = int(env.get("EMBEDDING_DIMENSIONS") or "1024")
        # 相似度阈值：低于它的检索结果直接判为"未找到"。
        # 向量检索的 top_k 永远会返回结果（哪怕毫不相关），不加阈值的话
        # 乱问的问题也会召回一条最高分的垃圾片段，Agent 容易拿去编答案。
        # 0.25 是保守值，可按真实 embedding 的分数分布用环境变量调整；设为 0 可关闭。
        threshold = float(env.get("EMBEDDING_SIMILARITY_THRESHOLD", "0.25"))
        timeout = float(env.get("EMBEDDING_TIMEOUT") or "60")
    except ValueError as e:
        raise EmbeddingConfigError(f"embedding 数值配置无法解析：{e}") from e
    return EmbeddingConfig(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        dimensions=dimensions,
        similarity_threshold=threshold,
        timeout=timeout,
    )


# 配置在 import 时解析一次；非法时不让 import 失败（服务照常启动），
# 而是记下错误，由 search / 建库 / 启动日志分别报告。
try:
    CONFIG: EmbeddingConfig | None = load_embedding_config(os.environ)
    CONFIG_ERROR: str | None = None
except EmbeddingConfigError as _e:
    CONFIG, CONFIG_ERROR = None, str(_e)

# 以下模块级常量保留旧名，供 evals / 测试读取
EMBEDDING_PROVIDER = CONFIG.provider if CONFIG else None
EMBEDDING_MODEL_NAME = CONFIG.model if CONFIG else os.getenv("EMBEDDING_MODEL_NAME")
SIMILARITY_THRESHOLD = CONFIG.similarity_threshold if CONFIG else 0.25

# 切分参数：chunk 太大则一段里混了多个主题，向量语义被稀释；
# 太小则上下文不完整。中文政策类文本一屏在 200-400 字，350 是经验值。
CHUNK_SIZE = 350
# 相邻 chunk 重叠的字符数：防止关键句子正好被切在边界上一分为二，
# 导致任何单个 chunk 都读不懂完整语义。
CHUNK_OVERLAP = 60

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
# 重试策略：只重试"连接 / 传输层"的瞬时故障（ConnectError、ReadError、RemoteProtocolError、
# ConnectTimeout 等，SSL: UNEXPECTED_EOF_WHILE_READING 就落在 ConnectError / ReadError 里）。
# 读超时 / 写超时 / 连接池超时不重试：对端慢时再等 3 轮只会把一次检索拖到几分钟。
# HTTP 4xx / 5xx 也不重试：那是请求或服务端的问题（key 错、模型名错、模型没 pull），重试没有意义。
RETRY_DELAYS = (1.0, 3.0)  # 第 1、2 次失败后的等待秒数 → 最多 3 次尝试
_NO_RETRY_ERRORS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)
# 测试注入点：_TRANSPORT 换成 httpx.MockTransport，_sleep 换成不真等的函数
_TRANSPORT: httpx.BaseTransport | None = None
_sleep = time.sleep
# 分批调用：embedding 接口对单次输入条数有限制，16 条一批是稳妥值
EMBED_BATCH_SIZE = 16


def _require_config() -> EmbeddingConfig:
    if CONFIG is None:
        raise EmbeddingConfigError(f"embedding 配置错误：{CONFIG_ERROR}")
    return CONFIG


def _host(url: str) -> str:
    """日志里只记 base_url 的主机部分。"""
    return urlsplit(url).netloc or url


def _post_json(url: str, payload: dict, headers: dict, stage: str, cfg: EmbeddingConfig) -> dict:
    """发一次 embedding 请求（带重试），返回解析后的 JSON。

    每次尝试都新建 Client（每次一个新连接），与原来直接调用 httpx.post 的连接行为一致。
    超时：总体沿用 EMBEDDING_TIMEOUT（默认 60s，与原来相同），只把 connect 收紧到 10s——
    连不上的时候尽快失败、进入重试，而不是在握手阶段干等 60s。
    日志只记 stage / provider / model / 主机 / 异常类型，不记 headers（里面有 key）。
    """
    attempts = len(RETRY_DELAYS) + 1
    started = time.monotonic()
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=httpx.Timeout(cfg.timeout, connect=10.0), transport=_TRANSPORT) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.TransportError as e:
            retryable = not isinstance(e, _NO_RETRY_ERRORS)
            detail = (
                f"stage={stage} provider={cfg.provider} model={cfg.model} host={_host(url)} "
                f"第 {attempt}/{attempts} 次 {type(e).__name__}: {str(e)[:200]} "
                f"已耗时 {time.monotonic() - started:.1f}s"
            )
            if retryable and attempt < attempts:
                delay = RETRY_DELAYS[attempt - 1]
                logger.warning("embedding 请求失败，%.0fs 后重试（%s）", delay, detail)
                _sleep(delay)
                continue
            reason = "重试次数用尽" if retryable else "超时类错误不重试"
            logger.error("embedding 请求失败，放弃：%s（%s）", reason, detail)
            raise
        if resp.status_code != 200:
            logger.error(
                "embedding 接口返回 HTTP %d（stage=%s provider=%s model=%s host=%s），不重试",
                resp.status_code, stage, cfg.provider, cfg.model, _host(url),
            )
            raise RuntimeError(
                f"embedding 接口返回 {resp.status_code}（model={cfg.model}，"
                f"base_url={cfg.base_url}）：{resp.text[:300]}"
            )
        return resp.json()
    raise AssertionError("unreachable")  # pragma: no cover


def embed_texts(texts: list[str], stage: str = "search") -> list[list[float]]:
    """批量向量化。stage 只用于日志（"build" 建库 / "search" 检索）。

    用 httpx 直接发 HTTP 请求而不是任何 SDK 封装，是因为两个接口本身都是简单的 POST：
    - zhipu：POST {base}/embeddings，请求体 {model, input, dimensions}，
      返回 {"data": [{"index", "embedding"}]}，按 index 排回原顺序；
    - ollama：POST {base}/api/embed，请求体 {model, input}，
      返回 {"embeddings": [[...], ...]}，顺序与 input 一致（bge-m3 为 1024 维）。
    这一层自己写，出问题时（换厂商、报错排查）每一行都是自己看得懂的。
    """
    cfg = _require_config()
    if cfg.provider == "zhipu":
        if not cfg.base_url or not cfg.api_key:
            raise RuntimeError(
                "缺少 embedding 配置：请在 .env 中配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY"
                "（智谱 embedding-3 需要智谱开放平台的 key）"
            )
        url = f"{cfg.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
    else:
        url = f"{cfg.base_url}/api/embed"
        headers = {}

    all_vectors: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        if cfg.provider == "zhipu":
            payload = {"model": cfg.model, "input": batch, "dimensions": cfg.dimensions}
            data = _post_json(url, payload, headers, stage, cfg)["data"]
            for item in data:  # 返回值带 index 字段，按它排回原顺序
                all_vectors[start + item["index"]] = item["embedding"]
        else:
            payload = {"model": cfg.model, "input": batch}
            embeddings = _post_json(url, payload, headers, stage, cfg)["embeddings"]
            if len(embeddings) != len(batch):
                raise RuntimeError(f"embedding 接口返回 {len(embeddings)} 条向量，期望 {len(batch)} 条")
            for i, vec in enumerate(embeddings):
                all_vectors[start + i] = vec

    if any(v is None for v in all_vectors):
        raise RuntimeError("embedding 接口返回的向量条数少于输入条数")
    if len({len(v) for v in all_vectors}) > 1:  # type: ignore[arg-type]
        raise RuntimeError("embedding 接口返回的向量维度不一致")
    return all_vectors  # type: ignore[return-value]


# ---------- 4. ChromaDB 集合 ----------
def get_collection():
    """获取（或创建）持久化的 ChromaDB 集合。

    hnsw:space 用 cosine 而不是默认的 l2：cosine 只关心向量方向（语义相似度），
    不关心模长，对文本 embedding 是标准选择。
    集合已存在时 get_or_create_collection 不会改动它已有的元数据（chromadb 1.5.9 实测）。
    """
    global _collection_cache
    if _collection_cache is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection_cache = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection_cache


# ---------- 5. 向量库与配置的一致性 ----------
# 建库时写进集合元数据的三个键：用哪个 provider、哪个模型、多少维建的库。
# 换了 embedding 却不重建，查询向量和库里的向量不在同一个语义空间里，
# 余弦相似度算出来是"有数字但没意义"——必须明确失败，而不是悄悄返回错的片段。
META_PROVIDER = "embedding_provider"
META_MODEL = "embedding_model"
META_DIMENSION = "embedding_dimension"
REBUILD_HINT = "需要重建：python scripts/build_kb.py --rebuild"

_legacy_warned = False


def index_metadata(cfg: EmbeddingConfig, dimension: int) -> dict:
    return {
        "hnsw:space": "cosine",
        META_PROVIDER: cfg.provider,
        META_MODEL: cfg.model,
        META_DIMENSION: dimension,
    }


def _stored_dimension(collection) -> int | None:
    got = collection.get(limit=1, include=["embeddings"])
    embeddings = got.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return len(embeddings[0])


def _mismatch_text(stored, cfg: EmbeddingConfig, query_dim: int | None = None) -> str:
    current = f"{cfg.provider}/{cfg.model}/{query_dim}" if query_dim is not None else cfg.describe()
    return (
        f"向量库与当前 embedding 配置不一致（库：{stored[0]}/{stored[1]}/{stored[2]}，当前：{current}）。"
        f"{REBUILD_HINT}"
    )


def _index_signature(collection, cfg: EmbeddingConfig) -> tuple:
    """返回库的 (provider, model, dimension)；provider / model 与当前配置不一致时抛 IndexMismatchError。

    没有元数据的旧库（引入本校验之前建的）：只有当前配置正是旧版默认的 zhipu/embedding-3/1024、
    且库里向量确实是 1024 维时才放行（写一次 WARNING 建议重建），其余情况一律按不一致处理。
    """
    global _legacy_warned
    meta = collection.metadata or {}
    if META_PROVIDER not in meta:
        stored_dim = _stored_dimension(collection)
        if (cfg.provider, cfg.model, cfg.dimensions) != LEGACY_DEFAULT or stored_dim != LEGACY_DEFAULT[2]:
            raise IndexMismatchError(
                f"向量库没有记录建库时的 embedding 配置（旧版建的库，向量维度 {stored_dim}），"
                f"无法确认与当前配置 {cfg.describe()} 一致。{REBUILD_HINT}"
            )
        if not _legacy_warned:
            logger.warning(
                "向量库没有记录建库时的 embedding 配置（旧版建的库），按 %s 放行；建议执行 %s 写入元数据",
                "/".join(map(str, LEGACY_DEFAULT)), "python scripts/build_kb.py --rebuild",
            )
            _legacy_warned = True
        return LEGACY_DEFAULT
    stored = (meta.get(META_PROVIDER), meta.get(META_MODEL), meta.get(META_DIMENSION))
    if (stored[0], stored[1]) != (cfg.provider, cfg.model):
        raise IndexMismatchError(_mismatch_text(stored, cfg))
    return stored


def _check_query_dim(stored, cfg: EmbeddingConfig, query_dim: int) -> None:
    if query_dim != stored[2]:
        raise IndexMismatchError(_mismatch_text(stored, cfg, query_dim))


def check_index(collection, cfg: EmbeddingConfig, query_dim: int | None = None) -> str | None:
    """一致返回 None，不一致返回原因（供建库 / 启动日志 / 测试使用）。

    query_dim 为 None 时只比较 provider / model（以及旧库规则）；
    维度要等拿到一条真实向量后才能比较（Ollama 的维度由模型决定，配置里没有）。
    """
    try:
        stored = _index_signature(collection, cfg)
        if query_dim is not None:
            _check_query_dim(stored, cfg, query_dim)
    except IndexMismatchError as e:
        return str(e)
    return None


def index_info() -> dict | None:
    """向量库元数据里记录的 provider / model / dimension（供评测结果留档）；读不到返回 None。"""
    try:
        meta = get_collection().metadata or {}
    except Exception:  # noqa: BLE001 只用于留档
        return None
    if META_PROVIDER not in meta:
        return None
    return {k: meta.get(k) for k in (META_PROVIDER, META_MODEL, META_DIMENSION)}


def log_index_status() -> str:
    """服务启动时调用一次：把配置与向量库的状态写进日志，返回状态名（便于测试）。

    ok / legacy（旧库放行）/ kb_not_built / mismatch / config_error / error
    """
    if CONFIG is None:
        logger.error("embedding 配置错误，知识库检索将失败：%s", CONFIG_ERROR)
        return "config_error"
    try:
        collection = get_collection()
        count = collection.count()
        if count == 0:
            logger.warning("知识库尚未建立（当前 embedding 配置 %s）", CONFIG.describe())
            return "kb_not_built"
        legacy = META_PROVIDER not in (collection.metadata or {})
        problem = check_index(collection, CONFIG)
    except Exception as e:  # noqa: BLE001 启动日志不能拖垮启动
        logger.error("检查向量库状态失败：%s: %s", type(e).__name__, e)
        return "error"
    if problem:
        logger.error("知识库检索将失败：%s", problem)
        return "mismatch"
    logger.info(
        "embedding 配置：%s，host=%s，相似度阈值 %s，向量库 %d 个 chunk%s",
        CONFIG.describe(), _host(CONFIG.base_url), SIMILARITY_THRESHOLD, count,
        "（旧版库，无元数据）" if legacy else "，与配置一致",
    )
    return "legacy" if legacy else "ok"


def configure_logging() -> None:
    """给入口脚本 / 服务用：让 rag 的 INFO 及以上日志输出到 stderr（不改 root logger）。"""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


# ---------- 6. 建库（幂等）----------
def build_knowledge_base(rebuild: bool = False) -> dict:
    """加载知识文档 → 切分 → 向量化 → 写入 ChromaDB。

    幂等逻辑：集合里已经有数据且未指定 rebuild 时直接跳过，
    所以重复运行、或者服务启动时误调用它，都不会重复入库。
    但如果已有数据是用另一套 embedding 配置建的，不跳过也不追加，而是报错要求 --rebuild。

    先向量化、成功后才删旧集合并新建：embedding 接口失败时旧库保持原样，不会被删空。

    返回统计信息 {files, chunks, skipped}。
    """
    cfg = _require_config()
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"knowledge/ 目录下没有找到任何 .md/.txt 文档：{KNOWLEDGE_DIR}")

    collection = get_collection()
    if collection.count() > 0 and not rebuild:
        problem = check_index(collection, cfg)
        if problem:
            raise IndexMismatchError(f"已有 {collection.count()} 个 chunk，拒绝在不一致的库上继续使用：{problem}")
        print(f"知识库已存在（{collection.count()} 个 chunk），跳过建库。如需重建：python scripts/build_kb.py --rebuild")
        return {"files": len(docs), "chunks": collection.count(), "skipped": True}

    # 所有 chunk 平铺，并记录来源文档，作为 metadata 存进去
    texts: list[str] = []
    metadatas: list[dict] = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})

    print(f"加载 {len(docs)} 篇文档，切分出 {len(texts)} 个 chunk，开始向量化（{cfg.describe()}）...")
    vectors = embed_texts(texts, stage="build")
    dimension = len(vectors[0])

    # 向量化成功后再删旧集合、按当前配置写元数据新建（重建时直接删集合最干净）
    global _collection_cache
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection(COLLECTION_NAME)
    _collection_cache = None
    collection = client.create_collection(name=COLLECTION_NAME, metadata=index_metadata(cfg, dimension))
    _collection_cache = collection

    # 批量写入。id 用 文件名::序号，既唯一又便于人工排查。
    collection.add(
        ids=[f"{m['source']}::{i}" for i, m in enumerate(metadatas)],
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"建库完成：{collection.count()} 个 chunk 已写入 {CHROMA_DIR}（{cfg.provider}/{cfg.model}/{dimension}）")
    return {"files": len(docs), "chunks": len(texts), "skipped": False}


# ---------- 7. 检索 ----------
def search(query: str, top_k: int = 3) -> str:
    """向量检索：query 向量化 → ChromaDB 余弦相似度 → top_k 片段。

    返回给上层的是一段带来源标注的纯文本上下文，Agent 拿到它后
    组织成自然语言回复。注意这个函数永远不抛异常：
    - 知识库没建 / 为空                → 提示先建库
    - 检索结果为空                     → "未找到相关信息"
    - embedding 接口挂了 / 配置错误     → 明确的失败说明（"知识库检索失败：…"）
    - 向量库与当前 embedding 配置不一致 → 同上，并提示 --rebuild
    这样 Agent 在任何情况下都能把结果继续往下说，而不是整条链路 500。
    """
    try:
        collection = get_collection()
        if collection.count() == 0:
            return KB_NOT_BUILT_MSG

        cfg = _require_config()
        # provider / model 不一致时在调用 embedding 之前就失败，不白发请求
        stored = _index_signature(collection, cfg)
        query_vector = embed_texts([query])[0]
        _check_query_dim(stored, cfg, len(query_vector))
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
        if isinstance(e, (IndexMismatchError, EmbeddingConfigError)):
            logger.error("知识库检索失败：%s", e)
        return f"知识库检索失败：{e}。请告知用户暂时无法查询知识库。"
