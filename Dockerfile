

FROM python:3.12-slim

WORKDIR /app

COPY benchmark/ /app/benchmark/

RUN pip install --no-cache-dir aiohttp pytest

CMD ["python", "benchmark/benchmark_qwen.py"]
