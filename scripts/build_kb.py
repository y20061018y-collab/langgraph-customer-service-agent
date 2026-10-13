"""
一键建库脚本：

    python scripts/build_kb.py            # 幂等建库，已有数据则跳过
    python scripts/build_kb.py --rebuild  # 删掉旧集合，全量重建

前置条件：.env 里配好 embedding 相关配置（默认用智谱 embedding-3；
EMBEDDING_PROVIDER=ollama 时用本地 Ollama，如 bge-m3）。

已有数据但与当前 embedding 配置不一致（换了 provider / 模型 / 维度）时，
不跳过也不追加，直接失败并提示加 --rebuild。
"""
import os
import sys

# 让脚本能从项目根目录 import rag（脚本不在根目录，需要手动加 path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import build_knowledge_base, configure_logging  # noqa: E402


def main():
    configure_logging()  # 让 embedding 重试 / 失败日志输出到 stderr
    rebuild = "--rebuild" in sys.argv
    try:
        stats = build_knowledge_base(rebuild=rebuild)
        print("统计信息:", stats)
    except RuntimeError as e:
        print(f"\n[建库失败] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
