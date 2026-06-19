import asyncio
import aiohttp
import time
import json
import sys

API_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = "Explain the importance of unit testing in software development."
MAX_TOKENS = 100
BATCH_SIZES = [1, 4, 16, 32]


async def send_request(session):
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
    }
    start = time.perf_counter()
    async with session.post(API_URL, json=payload) as resp:
        data = await resp.json()
        end = time.perf_counter()

    if "usage" not in data:
        raise RuntimeError(f"Unexpected response: {data}")

    latency = end - start
    completion_tokens = data["usage"]["completion_tokens"]
    return latency, completion_tokens


async def run_batch(batch_size):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session) for _ in range(batch_size)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end = time.perf_counter()

    total_wall_time = end - start
    latencies = sorted([r[0] for r in results])
    total_tokens = sum(r[1] for r in results)
    throughput = total_tokens / total_wall_time

    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]

    return {
        "model": MODEL_NAME,
        "batch_size": batch_size,
        "wall_time_sec": round(total_wall_time, 3),
        "total_tokens": total_tokens,
        "tokens_per_sec": round(throughput, 2),
        "p50_latency_sec": round(p50, 3),
        "p95_latency_sec": round(p95, 3),
        "p99_latency_sec": round(p99, 3),
    }


async def main():
    results = []
    for batch_size in BATCH_SIZES:
        print(f"Running batch size {batch_size}...")
        result = await run_batch(batch_size)
        results.append(result)
        print(json.dumps(result, indent=2))
        await asyncio.sleep(2)

    output_file = sys.argv[1] if len(sys.argv) > 1 else "qwen_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())