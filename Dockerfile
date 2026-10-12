FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 非 root 运行；chroma_db 目录预先建好，命名卷挂载时会继承属主
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/chroma_db \
    && chown -R app:app /app \
    && chmod +x docker/entrypoint.sh
USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["docker/entrypoint.sh"]
