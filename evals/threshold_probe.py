"""
相似度阈值校准：用校准集（evals/threshold_probe.json，不是验证集）测出当前 embedding 下
"知识库能回答"与"知识库没有答案"两类问题的 top-1 相似度分布，并按事先写定的规则给出阈值。

用法（项目根目录，需要先用同一套 embedding 配置建好知识库）：
    python evals/threshold_probe.py
    python evals/threshold_probe.py --probe evals/threshold_probe.json

做什么 / 不做什么：
- 直接 embed + collection.query 取 top-k，**不经过** rag.search 的阈值过滤，只记录原始相似度。
- 不调用对话模型，不写数据库。
- 运行前先做向量库一致性校验（rag.check_index），不一致直接退出，避免用错的库算出无意义的分布。
- 结果写到 evals/results/threshold_probe_<provider>-<model>_<时间>.json，不覆盖任何已有文件。

阈值选取规则（写在代码里，跑数据之前就固定，见 choose_threshold）：
A = 可回答问题的 top-1 相似度，U = 无答案问题的 top-1 相似度（业务相近 + 业务无关合并）。
- 可分（max U < min A）：T = 两者中点，四舍五入到 0.01；若取整后不再严格分开
  （不满足 max U < T <= min A），改为保留 4 位小数。
- 不可分：T = min A 向下取整到 0.01，即保住全部可回答问题、尽量多过滤无答案问题。
T 是在校准集内拟合出来的；保留率 / 过滤率只是对拟合结果的描述，不是独立的验证。
rag.search 按"相似度 < T 丢弃"逐个片段过滤，所以无答案问题被判为 not_found 当且仅当 top-1 < T。
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROBE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_probe.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

KINDS = ("answerable", "unanswerable")
GROUPS = ("near", "off")  # 无答案问题：业务相近 / 业务无关
STYLES = ("natural", "keyword")  # 自然问句 / 仿照 Agent 真实检索 query 的关键词短句


def describe(values: list[float]) -> dict:
    """n / 最小 / 中位 / 最大；空列表返回 n=0。"""
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def choose_threshold(answerable: list[float], unanswerable: list[float]) -> dict:
    """按模块说明里的规则选阈值，返回 {threshold, separable, rule}。"""
    if not answerable or not unanswerable:
        raise ValueError("两类问题都至少需要 1 条")
    min_a, max_u = min(answerable), max(unanswerable)
    if max_u < min_a:
        mid = (max_u + min_a) / 2
        t = round(mid, 2)
        rule = "可分：取中点，四舍五入到 0.01"
        if not (max_u < t <= min_a):
            t = round(mid, 4)
            rule = "可分：取中点（取整到 0.01 后不再严格分开，改为保留 4 位小数）"
        return {"threshold": t, "separable": True, "rule": rule}
    # round(…, 6) 去掉浮点误差（如 0.63 * 100 = 62.99999…），保证 T <= min A
    t = math.floor(round(min_a * 100, 6)) / 100
    return {"threshold": t, "separable": False, "rule": "不可分：min A 向下取整到 0.01（保住全部可回答问题）"}


def rate(values: list[float], pred) -> dict:
    hit = sum(1 for v in values if pred(v))
    return {"count": hit, "n": len(values), "rate": round(hit / len(values), 4) if values else None}


def analyze(cases: list[dict], top1: list[float]) -> dict:
    """cases 与 top1（每条的 top-1 相似度）一一对应，返回分布、阈值与在阈值下的保留 / 过滤情况。"""
    def pick(kind=None, group=None, style=None):
        return [
            s for c, s in zip(cases, top1)
            if (kind is None or c["kind"] == kind)
            and (group is None or c.get("group") == group)
            and (style is None or c["style"] == style)
        ]

    a_all, u_all = pick("answerable"), pick("unanswerable")
    choice = choose_threshold(a_all, u_all)
    t = choice["threshold"]
    distributions = {
        "answerable": describe(a_all),
        "unanswerable": describe(u_all),
        "unanswerable_near": describe(pick("unanswerable", "near")),
        "unanswerable_off": describe(pick("unanswerable", "off")),
    }
    for style in STYLES:
        distributions[f"answerable_{style}"] = describe(pick("answerable", style=style))
        distributions[f"unanswerable_{style}"] = describe(pick("unanswerable", style=style))
    at_threshold = {
        # 保留率只描述拟合结果（不可分时按规则必为 100%），不作为通过条件
        "answerable_kept": rate(a_all, lambda v: v >= t),
        "unanswerable_near_filtered": rate(pick("unanswerable", "near"), lambda v: v < t),
        "unanswerable_off_filtered": rate(pick("unanswerable", "off"), lambda v: v < t),
        "unanswerable_filtered": rate(u_all, lambda v: v < t),
    }
    for style in STYLES:
        at_threshold[f"answerable_{style}_kept"] = rate(pick("answerable", style=style), lambda v: v >= t)
        at_threshold[f"unanswerable_{style}_filtered"] = rate(pick("unanswerable", style=style), lambda v: v < t)
    return {
        "threshold": t,
        "separable": choice["separable"],
        "rule": choice["rule"],
        "distributions": distributions,
        "at_threshold": at_threshold,
    }


def validate_cases(cases: list[dict], known_sources: set[str]) -> None:
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("校准集里有重复 id")
    for c in cases:
        if c["kind"] not in KINDS or c["style"] not in STYLES:
            raise ValueError(f"{c['id']}：kind / style 取值非法")
        if c["kind"] == "answerable":
            if c.get("expected_source") not in known_sources:
                raise ValueError(f"{c['id']}：expected_source 不在 knowledge/ 中：{c.get('expected_source')}")
        elif c.get("group") not in GROUPS:
            raise ValueError(f"{c['id']}：无答案问题的 group 必须是 {GROUPS}")


def main():
    import rag

    parser = argparse.ArgumentParser(description="相似度阈值校准（校准集，不是验证集）")
    parser.add_argument("--probe", default=PROBE_FILE, help="校准集文件，默认 evals/threshold_probe.json")
    parser.add_argument("--top-k", type=int, default=3, help="每条记录前几个片段，默认 3（与 rag.search 一致）")
    args = parser.parse_args()
    rag.configure_logging()

    with open(args.probe, "rb") as f:
        raw = f.read()
    cases = json.loads(raw.decode("utf-8"))["cases"]
    validate_cases(cases, {d["source"] for d in rag.load_documents()})

    cfg = rag._require_config()  # noqa: SLF001
    collection = rag.get_collection()
    count = collection.count()
    if count == 0:
        print(rag.KB_NOT_BUILT_MSG)
        sys.exit(1)
    problem = rag.check_index(collection, cfg)
    if problem:
        print(f"[校准中止] {problem}")
        sys.exit(1)

    vectors = rag.embed_texts([c["query"] for c in cases], stage="probe")
    problem = rag.check_index(collection, cfg, query_dim=len(vectors[0]))
    if problem:
        print(f"[校准中止] {problem}")
        sys.exit(1)
    result = collection.query(
        query_embeddings=vectors,
        n_results=min(args.top_k, count),
        include=["metadatas", "distances"],
    )

    records, top1 = [], []
    for c, metas, dists in zip(cases, result["metadatas"], result["distances"]):
        hits = [{"source": m.get("source"), "similarity": round(1 - d, 4)} for m, d in zip(metas, dists)]
        sources = [h["source"] for h in hits]
        rec = {**c, "hits": hits, "top1_similarity": hits[0]["similarity"], "top1_source": hits[0]["source"]}
        if c["kind"] == "answerable":
            rec["top1_correct"] = hits[0]["source"] == c["expected_source"]
            rec["expected_rank"] = sources.index(c["expected_source"]) + 1 if c["expected_source"] in sources else None
        records.append(rec)
        top1.append(1 - dists[0])  # 用未取整的值做统计和选阈值

    summary = analyze(cases, top1)
    answerable = [r for r in records if r["kind"] == "answerable"]
    summary["answerable_top1_correct"] = rate([1.0 if r["top1_correct"] else 0.0 for r in answerable], lambda v: v == 1.0)

    started = datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    label = re.sub(r"[^A-Za-z0-9._-]", "-", f"{cfg.provider}-{cfg.model}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"threshold_probe_{label}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "time": stamp,
                    "embedding_provider": cfg.provider,
                    "embedding_model": cfg.model,
                    "embedding_index": rag.index_info(),
                    "query_dimension": len(vectors[0]),
                    "chunks": count,
                    "top_k": min(args.top_k, count),
                    "probe_file": os.path.basename(args.probe),
                    "probe_sha256": hashlib.sha256(raw).hexdigest(),
                    "cases": len(cases),
                },
                "summary": summary,
                "results": records,
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"embedding：{cfg.provider}/{cfg.model}（{len(vectors[0])} 维），{count} 个 chunk，{len(cases)} 条校准问题")
    for name, d in summary["distributions"].items():
        print(f"  {name:24s} n={d['n']:<3} min={d['min']}  median={d['median']}  max={d['max']}")
    print(f"可分：{summary['separable']}；阈值 T = {summary['threshold']}（{summary['rule']}）")
    for name, r in summary["at_threshold"].items():
        print(f"  {name:32s} {r['count']}/{r['n']}")
    print(f"可回答问题 top-1 来源正确：{summary['answerable_top1_correct']['count']}/{len(answerable)}")
    print(f"结果已保存：{out}")


if __name__ == "__main__":
    main()
