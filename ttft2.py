import time
import json
import requests

URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
PROMPT = "What is DevOps?"

payload = {
    "model": MODEL,
    "prompt": PROMPT,
    "stream": True
}

start = time.perf_counter()

r = requests.post(URL, json=payload, stream=True)
r.raise_for_status()

first_token = None
tokens = 0

for line in r.iter_lines():
    if not line:
        continue

    data = json.loads(line)

    if first_token is None and data.get("response"):
        first_token = time.perf_counter()

    if "eval_count" in data:
        tokens = data["eval_count"]

end = time.perf_counter()

ttft = first_token - start
total = end - start
token_sec = tokens / (end - first_token)

print(f"TTFT      : {ttft:.3f} sec")
print(f"Tokens    : {tokens}")
print(f"Token/sec : {token_sec:.2f}")
print(f"Total     : {total:.3f} sec")
