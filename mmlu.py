import os 
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models import OllamaModel

os.environ["OLLAMA_BASE_API"] = "http://34.57.232.168:11434"
local_model=OllamaModel(model="llama3.2:latest")

benchmark = MMLU(
    tasks=[MMLUTask.COLLEGE_COMPUTER_SCIENCE],
    n_shots=1
)

benchmark.evaluate(model=local_model)

print("This is the result : ",benchmark.overall_score)