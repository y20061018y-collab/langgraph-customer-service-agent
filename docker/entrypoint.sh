#!/bin/sh
set -e

# 1. 灌入模拟订单（幂等 upsert，重复启动不会产生重复数据）
python scripts/seed_data.py

# 2. 建知识库（幂等：chroma_db 卷里已有数据则跳过）
#    需要 embedding key；失败不阻塞启动，检索会降级返回"知识库尚未建立"
python scripts/build_kb.py || echo "[warn] 知识库构建失败，检查 .env 中的 EMBEDDING_* 配置后执行：docker compose exec app python scripts/build_kb.py"

# 3. 启动 API
exec uvicorn main:app --host 0.0.0.0 --port 8000
