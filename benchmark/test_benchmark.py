import json
import os


def test_results_file_structure():
    # checks that a sample results file (if present) has the expected keys
    sample_path = os.path.join(os.path.dirname(__file__), "qwen_results.json")
    if not os.path.exists(sample_path):
        return

    with open(sample_path) as f:
        data = json.load(f)

    required_keys = {
        "batch_size",
        "wall_time_sec",
        "total_tokens",
        "tokens_per_sec",
        "p50_latency_sec",
        "p95_latency_sec",
        "p99_latency_sec",
    }

    for entry in data:
        assert required_keys.issubset(entry.keys())
        assert entry["tokens_per_sec"] >= 0
        assert entry["batch_size"] in [1, 4, 16, 32]


def test_throughput_calculation():
    total_tokens = 400
    wall_time_sec = 2.0
    expected_throughput = total_tokens / wall_time_sec
    assert expected_throughput == 200.0


def test_percentile_ordering():
    latencies = sorted([0.5, 0.6, 0.7, 0.8, 0.9])
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    assert p50 <= p95
