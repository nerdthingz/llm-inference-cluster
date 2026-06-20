import asyncio
import aiohttp
import time
import json
from collections import Counter

NGINX_URL = "http://localhost:8080/v1/completions"
NUM_REQUESTS = 30
PROMPT = "What are the benefits of regular exercise?"
MAX_TOKENS = 50

# both models must be loaded for this to work since nginx routes to either
MODEL_BY_BACKEND = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "tinyllama": "Qwen/Qwen2.5-0.5B-Instruct",
}


async def send_request(session, request_id):
    # nginx picks the backend, so we try qwen's model name first
    # if nginx routes to the small model it will reject the wrong model name,
    # so we detect via the X-Backend-Used header and retry with correct name if needed
    payload = {
        "model": MODEL_BY_BACKEND["qwen"],
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
    }
    start = time.perf_counter()
    async with session.post(NGINX_URL, json=payload) as resp:
        backend = resp.headers.get("X-Backend-Used", "unknown")
        data = await resp.json()

        if "error" in data and backend == "tinyllama":
            payload["model"] = MODEL_BY_BACKEND["tinyllama"]
            start = time.perf_counter()
            async with session.post(NGINX_URL, json=payload) as resp2:
                backend = resp2.headers.get("X-Backend-Used", backend)
                data = await resp2.json()

        end = time.perf_counter()

    latency = end - start
    tokens = data.get("usage", {}).get("completion_tokens", 0)
    return {"request_id": request_id, "backend": backend, "latency": latency, "tokens": tokens}


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, i) for i in range(NUM_REQUESTS)]
        results = await asyncio.gather(*tasks)

    backend_counts = Counter(r["backend"] for r in results)
    print(f"\nTotal requests: {NUM_REQUESTS}")
    print(f"Backend distribution: {dict(backend_counts)}")
    print("Expected ~80/20 split (qwen/tinyllama)\n")

    for backend in backend_counts:
        backend_results = [r for r in results if r["backend"] == backend]
        avg_latency = sum(r["latency"] for r in backend_results) / len(backend_results)
        avg_tokens = sum(r["tokens"] for r in backend_results) / len(backend_results)
        print(f"{backend}: avg_latency={avg_latency:.3f}s, avg_tokens={avg_tokens:.1f}, count={len(backend_results)}")

    with open("ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to ab_test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
