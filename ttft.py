import time
import requests
import statistics

OLLAMA_URL = "http://34.57.232.168/ollama/api/generate"

MODEL = "qwen3:8b"

PROMPT = """
Explain what Kubernetes is and how a Kubernetes cluster works.
Explain the control plane, worker nodes, pods, services, kubelet,
kube-proxy, and container runtime in a clear technical way.
"""

NUM_RUNS = 1


def benchmark_once():
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": True
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True
    )

    response.raise_for_status()

    first_token_time = None
    full_response = ""

    for line in response.iter_lines():
        if not line:
            continue

        data = line.decode("utf-8")

        # Ollama returns JSON per line
        import json
        chunk = json.loads(data)

        token = chunk.get("response", "")
        full_response += token

        # First token received
        if first_token_time is None and token:
            first_token_time = time.perf_counter()

    end_time = time.perf_counter()

    # Timing
    ttft = first_token_time - start_time
    total_time = end_time - start_time
    generation_time = end_time - first_token_time

    # Ollama provides eval_count
    token_count = chunk.get("eval_count", 0)

    # Ollama's eval_duration is nanoseconds
    eval_duration = chunk.get("eval_duration", 0)

    if eval_duration > 0:
        tokens_per_second = token_count / (eval_duration / 1e9)
    else:
        tokens_per_second = token_count / generation_time

    return {
        "ttft": ttft,
        "total_time": total_time,
        "generation_time": generation_time,
        "tokens": token_count,
        "tokens_per_second": tokens_per_second,
        "response": full_response
    }


def main():
    print("=" * 60)
    print("Ollama LLM Benchmark")
    print("=" * 60)

    print(f"Model : {MODEL}")
    print(f"Runs  : {NUM_RUNS}")
    print()

    results = []

    for i in range(NUM_RUNS):
        print(f"Running benchmark {i + 1}/{NUM_RUNS}...")

        result = benchmark_once()
        results.append(result)

        print(f"TTFT       : {result['ttft']:.4f} sec")
        print(f"Tokens     : {result['tokens']}")
        print(f"Token/s    : {result['tokens_per_second']:.2f}")
        print(f"Generation : {result['generation_time']:.4f} sec")
        print(f"Total      : {result['total_time']:.4f} sec")
        print()

    # Statistics
    ttfts = [r["ttft"] for r in results]
    tok_s = [r["tokens_per_second"] for r in results]
    totals = [r["total_time"] for r in results]

    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Average TTFT      : {statistics.mean(ttfts):.4f} sec")
    print(f"Min TTFT          : {min(ttfts):.4f} sec")
    print(f"Max TTFT          : {max(ttfts):.4f} sec")
    print()

    print(f"Average Token/s   : {statistics.mean(tok_s):.2f}")
    print(f"Min Token/s       : {min(tok_s):.2f}")
    print(f"Max Token/s       : {max(tok_s):.2f}")
    print()

    print(f"Average Total     : {statistics.mean(totals):.4f} sec")

    print("=" * 60)


if __name__ == "__main__":
    main()