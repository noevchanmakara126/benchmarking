import time 
import tiktoken
from openai import OpenAI 
client= OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
model_name = "qwen3:8b"
try:
    encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoder = None
    print("Warning: tiktoken not found. Falling back to character estimation.")
PROMPTS = [
    {"name": "Code: Refactor", "category": "coding", "prompt": "Refactor this Python function to be more concise:\n\ndef find_max(numbers):\n    max_val = numbers[0]\n    for n in numbers:\n        if n > max_val:\n            max_val = n\n    return max_val"},
    {"name": "Code: FastAPI", "category": "coding", "prompt": "Write a complete FastAPI endpoint for handling user registration with Pydantic validation and password hashing using passlib."},
    {"name": "Reasoning: Race", "category": "reasoning", "prompt": "If you overtake the second person in a race, what's your position? Where is the person you overtook?"},
    {"name": "Math: Geometry", "category": "math", "prompt": "Calculate the area of a triangle with vertices at (0, 0), (-1, 1), and (3, 3)."},
    {"name": "Writing: Email", "category": "writing", "prompt": "Write a professional email to your supervisor seeking feedback on the Quarterly Financial Report. Ask specifically about data analysis, presentation style, and clarity of conclusions. Keep it concise."},
]


print("=" * 80)
print(f"STARTING BENCHMARK LOOP FOR MODEL: {model_name}")
print("=" * 80)


#warm up first 
print("Warming up model (loading into memory)...")
_ = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=5,
    temperature=0.0
)
print("Warm-up complete. Starting benchmark...\n")

results = []
total_tokens = 0
total_gen_time = 0.0

for item in PROMPTS :
    propmt_text = item["prompt"]
    propmt_name = item["name"]
    category = item["category"]

    start_time = time.perf_counter()
    first_token_time = None
    text_chunks = []

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": propmt_text}],
        stream=True,
        temperature=0.1,   
       # max_tokens=1024
    )
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
           text_chunk = chunk.choices[0].delta.content
           if first_token_time is None:
              first_token_time = time.perf_counter()
           text_chunks.append(text_chunk)
    end_time = time.perf_counter()
    generate_text = "".join(text_chunk)

    if encoder:
        actual_tokens = len(encoder.encode(generate_text))
    else:
        actual_tokens = max(1,len(generate_text) // 4)
    total_tokens += actual_tokens


    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    gen_time = end_time - first_token_time if first_token_time else 0 
    tok_per_sec = actual_tokens / gen_time if gen_time > 0 else 0 

    results.append({
        "name" : propmt_name,
        "category" : category,
        "ttft_ms" : ttft_ms,
        "tokens" : actual_tokens,
        "tok_per_sec" : tok_per_sec
    })
    print(f"[{category.upper()}] {propmt_name:<20} | TTFT: {ttft_ms:6.2f} ms | Speed: {tok_per_sec:6.2f} tok/s | Tokens: {actual_tokens}")
avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
aggrefate_throughput = total_tokens / total_gen_time if total_gen_time > 0 else 0

print("=" * 80)
print("FINAL BENCHMARK SUMMARY")
print("=" * 80)
print(f"Prompts Evaluated:  {len(results)}")
print(f"Average TTFT:       {avg_ttft:.2f} ms")
print(f"Average Throughput: {aggrefate_throughput:.2f} tok/s")
print("=" * 80)