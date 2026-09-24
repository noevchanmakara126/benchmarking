import os 
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import OllamaModel

os.environ["OLLAMA_BASE_API"] = "http://localhost:11434"
local_model=OllamaModel(model="llama3.2:latest")

benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_MATHEMATICS, MMLUTask.ASTRONOMY],
    n_shots=1
)

benchmark.evaluate(model=local_model)

print("This is the result : ",benchmark.overall_score)