#!/usr/bin/env python3
"""
Simple local LLM coding benchmark for Ollama.

Requirements:
  1. Install Ollama: https://ollama.com
  2. Pull a model, e.g.:
       ollama pull llama3.2:3b
  3. Make sure Ollama is running.
  4. Run:
       python3 benchmark.py
     or:
       python3 benchmark.py qwen3:4b

The benchmark uses deterministic coding tasks and executes the generated
Python code against tests. It is intentionally simple for classroom use.
"""

import json
import re
import sys
import time
import subprocess
from urllib.request import Request, urlopen
from urllib.error import URLError

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"

# Each task asks the model to return ONLY Python code.
# The test cases are executed locally, so the score is based on correctness.
TASKS = [

    {
        "name": "Sum of Digits",
        "prompt": """Write a Python function named sum_digits(n) that returns the
sum of all digits in a non-negative integer.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert sum_digits(1234) == 10
assert sum_digits(999) == 27
assert sum_digits(0) == 0
assert sum_digits(10001) == 2
""",
    },

    {
        "name": "Reverse Words",
        "prompt": """Write a Python function named reverse_words(s) that returns
the words in the string in reverse order.

Remove extra spaces between words and at the beginning or end.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert reverse_words("hello world") == "world hello"
assert reverse_words("  hello   world  ") == "world hello"
assert reverse_words("one two three") == "three two one"
assert reverse_words("hello") == "hello"
""",
    },

    {
        "name": "Find Missing Number",
        "prompt": """Write a Python function named find_missing(numbers) that
takes a list containing distinct integers from 0 through n with exactly one
number missing. Return the missing number.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert find_missing([3, 0, 1]) == 2
assert find_missing([0, 1]) == 2
assert find_missing([9, 6, 4, 2, 3, 5, 7, 0, 1]) == 8
assert find_missing([0]) == 1
""",
    },

    {
        "name": "Move Zeroes",
        "prompt": """Write a Python function named move_zeroes(numbers) that
moves all zero values to the end of the list while preserving the relative
order of all non-zero values.

Return a new list.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert move_zeroes([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0]
assert move_zeroes([1, 2, 3]) == [1, 2, 3]
assert move_zeroes([0, 0, 1]) == [1, 0, 0]
assert move_zeroes([]) == []
""",
    },

    {
        "name": "Intersection Lists",
        "prompt": """Write a Python function named intersection(a, b) that
returns a list containing the unique values that appear in both lists.
Preserve the order in which values first appear in list a.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert intersection([1, 2, 2, 3], [2, 3, 4]) == [2, 3]
assert intersection(["a", "b", "c"], ["b", "c"]) == ["b", "c"]
assert intersection([1, 2, 3], [4, 5]) == []
assert intersection([], [1, 2]) == []
""",
    },

    {
        "name": "Character Frequency",
        "prompt": """Write a Python function named char_frequency(s) that returns
a dictionary containing the frequency of every character in the string.
Treat uppercase and lowercase characters as different characters.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert char_frequency("hello") == {"h": 1, "e": 1, "l": 2, "o": 1}
assert char_frequency("aaa") == {"a": 3}
assert char_frequency("") == {}
assert char_frequency("aA") == {"a": 1, "A": 1}
""",
    },

    {
        "name": "First Unique Character",
        "prompt": """Write a Python function named first_unique_char(s) that
returns the index of the first character that appears exactly once.
Return -1 if no unique character exists.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert first_unique_char("leetcode") == 0
assert first_unique_char("loveleetcode") == 2
assert first_unique_char("aabb") == -1
assert first_unique_char("abcabc") == -1
""",
    },

    {
        "name": "Merge Intervals",
        "prompt": """Write a Python function named merge_intervals(intervals)
that merges all overlapping intervals.

For example, [[1,3], [2,6], [8,10]] becomes
[[1,6], [8,10]].

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]
assert merge_intervals([]) == []
""",
    },

    {
        "name": "Maximum Subarray",
        "prompt": """Write a Python function named max_subarray(numbers) that
returns the largest possible sum of a contiguous subarray.

The input contains at least one number.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5,4,-1,7,8]) == 23
assert max_subarray([-5,-2,-8]) == -2
""",
    },

    {
        "name": "Valid Anagram",
        "prompt": """Write a Python function named valid_anagram(s, t) that
returns True if s and t contain exactly the same characters with the same
frequencies. Otherwise return False.

Treat uppercase and lowercase as different characters.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert valid_anagram("anagram", "nagaram") is True
assert valid_anagram("rat", "car") is False
assert valid_anagram("", "") is True
assert valid_anagram("aabb", "bbaa") is True
""",
    },

    {
        "name": "Climbing Stairs",
        "prompt": """Write a Python function named climb_stairs(n) that returns
the number of distinct ways to climb n stairs when you can climb either
1 or 2 stairs at a time.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert climb_stairs(1) == 1
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(5) == 8
assert climb_stairs(10) == 89
""",
    },

    {
        "name": "Power of Two",
        "prompt": """Write a Python function named is_power_of_two(n) that
returns True if n is a power of two, otherwise False.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert is_power_of_two(1) is True
assert is_power_of_two(2) is True
assert is_power_of_two(8) is True
assert is_power_of_two(16) is True
assert is_power_of_two(18) is False
assert is_power_of_two(0) is False
""",
    },

    {
        "name": "GCD",
        "prompt": """Write a Python function named gcd(a, b) that returns the
greatest common divisor of two non-negative integers.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert gcd(48, 18) == 6
assert gcd(17, 5) == 1
assert gcd(100, 25) == 25
assert gcd(0, 5) == 5
assert gcd(7, 0) == 7
""",
    },

    {
        "name": "Prime Number",
        "prompt": """Write a Python function named is_prime(n) that returns
True if n is a prime number, otherwise False.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert is_prime(2) is True
assert is_prime(3) is True
assert is_prime(4) is False
assert is_prime(17) is True
assert is_prime(25) is False
assert is_prime(1) is False
assert is_prime(0) is False
""",
    },

    {
        "name": "Longest Common Prefix",
        "prompt": """Write a Python function named longest_common_prefix(strings)
that returns the longest common prefix shared by all strings in the list.
Return an empty string if there is no common prefix.

Return ONLY Python code. Do not use markdown fences.""",
        "tests": """
assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
assert longest_common_prefix(["dog", "racecar", "car"]) == ""
assert longest_common_prefix(["interspecies", "interstellar", "interstate"]) == "inters"
assert longest_common_prefix(["hello"]) == "hello"
assert longest_common_prefix([]) == ""
""",
    },

]


def call_ollama(model, prompt):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }).encode()

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode())
    elapsed = time.perf_counter() - start

    return data["response"], elapsed, data


def clean_code(text):
    # Remove markdown fences if the model ignores the instruction.
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def run_tests(generated_code, tests):
    # Separate namespace prevents test variables leaking between tasks.
    namespace = {}

    full_code = generated_code + "\n\n" + tests

    try:
        exec(full_code, namespace)
        return True, "All tests passed"
    except AssertionError:
        return False, "A test assertion failed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print("=" * 60)
    print("        LOCAL LLM CODING BENCHMARK")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Tasks: {len(TASKS)}")
    print("Temperature: 0")
    print()

    # Check whether Ollama is reachable.
    try:
        with urlopen("http://localhost:11434/api/tags", timeout=5) as response:
            tags = json.loads(response.read().decode())
        installed = [m["name"] for m in tags.get("models", [])]
        if not any(model == x or model.split(":")[0] == x.split(":")[0] for x in installed):
            print(f"WARNING: {model} may not be pulled locally.")
            print(f"Try: ollama pull {model}")
            print()
    except Exception:
        print("ERROR: Cannot connect to Ollama at localhost:11434.")
        print("Start Ollama and try again.")
        sys.exit(1)

    passed = 0
    total_time = 0.0
    total_generated_tokens = 0

    results = []

    for i, task in enumerate(TASKS, start=1):
        print(f"[{i:02d}/{len(TASKS)}] {task['name']} ... ", end="", flush=True)

        try:
            output, elapsed, raw = call_ollama(model, task["prompt"])
            total_time += elapsed

            # Ollama reports eval_count = generated tokens.
            total_generated_tokens += raw.get("eval_count", 0)

            code = clean_code(output)
            ok, message = run_tests(code, task["tests"])

            if ok:
                passed += 1
                print(f"PASS  ({elapsed:.2f}s)")
            else:
                print(f"FAIL  ({elapsed:.2f}s) - {message}")

            results.append({
                "task": task["name"],
                "passed": ok,
                "time_seconds": round(elapsed, 3),
                "message": message,
                "code": code,
            })

        except Exception as exc:
            print(f"ERROR - {exc}")
            results.append({
                "task": task["name"],
                "passed": False,
                "time_seconds": 0,
                "message": str(exc),
                "code": "",
            })

    score = (passed / len(TASKS)) * 100
    avg_time = total_time / len(TASKS)
    tok_per_sec = total_generated_tokens / total_time if total_time else 0

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Passed:              {passed}/{len(TASKS)}")
    print(f"Failed:              {len(TASKS) - passed}/{len(TASKS)}")
    print(f"Score:               {score:.1f}%")
    print(f"Total generation:    {total_time:.2f}s")
    print(f"Average task time:   {avg_time:.2f}s")
    print(f"Generated tokens:    {total_generated_tokens}")
    print(f"Generation speed:    {tok_per_sec:.1f} tokens/sec")
    print("=" * 60)

    # Save results so students can inspect exactly what happened.
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": model,
            "tasks": len(TASKS),
            "passed": passed,
            "score_percent": round(score, 2),
            "total_time_seconds": round(total_time, 3),
            "average_task_time_seconds": round(avg_time, 3),
            "generated_tokens": total_generated_tokens,
            "tokens_per_second": round(tok_per_sec, 2),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print("\nDetailed results saved to: benchmark_results.json")


if __name__ == "__main__":
    main()
