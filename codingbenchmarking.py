import time
import re
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/chat",
    api_key="ollama"
)

model_name = "qwen3:8b"

# 1. Dataset of Coding Tasks with Unit Assertions
CODING_DATASET = [
    {
        "id": "task_1",
        "prompt": "Write a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` that checks if in given list of numbers, any two numbers are closer to each other than the given threshold.",
        "test_cases": """
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
"""
    },
    {
        "id": "task_2",
        "prompt": "Write a Python function `separate_paren_groups(paren_string: str) -> list[str]` that separates nested parentheses into distinct strings. Strip spaces.",
        "test_cases": """
assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
"""
    },
    {
        "id": "task_3",
        "prompt": "Write a Python function `is_prime(n: int) -> bool` that returns True if a number is prime, else False.",
        "test_cases": """
assert is_prime(6) == False
assert is_prime(101) == True
assert is_prime(1) == False
"""
    }
]

def extract_code(text: str) -> str:
    """Extracts raw Python code block from LLM response markdown."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

def evaluate_code(generated_code: str, test_cases: str) -> bool:
    """Executes code and assertions in a clean global scope."""
    combined = f"{generated_code}\n\n# --- UNIT TESTS ---\n{test_cases}"
    try:
        exec_globals = {}
        exec(combined, exec_globals)
        return True
    except Exception:
        return False

# 2. Run Benchmark
print("=" * 60)
print(f"RUNNING CODING EVALUATION (PASS@1) FOR: {model_name}")
print("=" * 60)

passed = 0
total = len(CODING_DATASET)

for task in CODING_DATASET:
    start = time.perf_counter()
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": task["prompt"]}],
        temperature=0.0
    )
    
    duration = time.perf_counter() - start
    output_text = response.choices[0].message.content
    code_only = extract_code(output_text)
    
    is_correct = evaluate_code(code_only, task["test_cases"])
    if is_correct:
        passed += 1
        status = "PASSED"
    else:
        status = "FAILED"
        
    print(f"Task ID: {task['id']:<10} | Result: {status:<6} | Execution Time: {duration:.2f}s")

# 3. Final Summary Report
pass_rate = (passed / total) * 100
print("=" * 60)
print("CODING BENCHMARK SCORE SUMMARY")
print("=" * 60)
print(f"Total Tasks:  {total}")
print(f"Tasks Passed: {passed}")
print(f"Pass@1 Score: {pass_rate:.1f}%")
print("=" * 60)