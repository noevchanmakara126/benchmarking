import os
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask
from deepeval.models import OllamaModel

os.environ["OLLAMA_BASE_API"] = "http://localhost:11434"
local_model=OllamaModel(model="gwen3:8b")

benchmark = HumanEval(
    tasks=[HumanEvalTask.STRING_XOR],
    n=1
)

benchmark.evaluate(model=local_model,k=1)

print(benchmark.overall_score)