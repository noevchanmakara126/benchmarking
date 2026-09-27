import datasets

_original_load_dataset = datasets.load_dataset

def _patched_load_dataset(path, *args, **kwargs):
    if path == "openai_humaneval":
        path = "openai/openai_humaneval"
    return _original_load_dataset(path, *args, **kwargs)

datasets.load_dataset = _patched_load_dataset


import re
import ollama
from deepeval.models import DeepEvalBaseLLM
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask

OLLAMA_HOST = "http://localhost:11434"


class QwenModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "qwen3:8b", host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def load_model(self):
        return self.client

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        m = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
        return (m.group(1) if m else text).strip()

    def generate(self, prompt: str) -> str:
        r = self.client.generate(model=self.model_name, prompt=prompt, think=False)
        return self._clean(r["response"])

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate_samples(self, prompt: str, n: int, temperature: float) -> list[str]:
        samples = []
        for _ in range(n):
            r = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                think=False,
                options={"temperature": temperature},
            )
            samples.append(self._clean(r["response"]))
        return samples

    def get_model_name(self):
        return self.model_name


qwen = QwenModel()

benchmark = HumanEval(tasks=[HumanEvalTask.PARSE_MUSIC], n=1)
benchmark.evaluate(model=qwen, k=1)
print(benchmark.overall_score)