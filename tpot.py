import time
import json
import requests 


while True : 
 MODEL_URL = input("Input Model URL: ") # http://34.57.232.168:11434/api/generate
 LOCAL_MODEL = input("Input Model: ")
 PROMPT = input("Propmt : ")

 payload = {
    "model" : LOCAL_MODEL,
    "prompt" : PROMPT,
    "stream" : True
 }



 first_token = None 
 total_tokens = 0


 print("Your model is :",LOCAL_MODEL)
 print("Your model url is : ",MODEL_URL)
 print("-"*50)

 start_time = time.perf_counter()
 res = requests.post(
  MODEL_URL,
  json=payload,
  stream=True)

 res.raise_for_status()
 for line in res.iter_lines():
  if not line:
   continue
  data = json.loads(line)
  if first_token is None and data.get("response"):
   first_token = time.perf_counter()
  if "eval_count" in data:
   total_token= data["eval_count"]

#calculation

 end_time = time.perf_counter()
 total_time = end_time - start_time
 generated_time = end_time - first_token
 tpot = generated_time / total_token

 print(f"TPOT           : {tpot:.4f} sec/token")
 print(f"TPOT           : {tpot * 1000:.2f} ms/token")
 print(f"Tokens         : {total_token}")
 print(f"Total Time     : {total_time:.3f} sec")
 print("-"*50)

   
 answer = input("Do you want to benchmark again? [yes/no] :")
 if answer == 'no':
    end_time = 0
    total_time =  0
    total_token = 0
    generated_time = 0
    tpot = 0
    print("Goodbye!")
    break  