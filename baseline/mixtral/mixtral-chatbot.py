from datetime import datetime
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# add the path to the sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.evaluate.Perplexity import Perplexity

# relative path to this file
mistral_models_path = os.path.join(os.path.dirname(__file__), "../../model/llm/")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                          cache_dir=mistral_models_path, device_map='auto', torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
                                             cache_dir=mistral_models_path, device_map='auto', torch_dtype=torch.float16)

# to disable a warning: Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
model.generation_config.pad_token_id = tokenizer.pad_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_new_tokens = 10000


i = 0
messages = []
while True:
    # if i > 0:
    #     msg_text = input(f"User {i}: ")
    # else:
    #     msg_text = '''
    #         You are a data analysis assistant who strictly adheres to instructions. You have access to a adult salary model which can predict if a person's salary is above 50K. Your task is divided into three steps:
            
    #         The json description of the model's input fields is as follows:
            
    #         {
    #             "X": {
    #                 "age": {"type": "int", "range": [17, 100]},
    #                 "capital-gain": {"type": "int", "range": [0, 100000]},
    #                 "capital-loss": {"type": "int", "range": [0, 100000]},
    #                 "education-num": {"type": "int", "range": [1, 16]},
    #                 "education": {
    #                     "type": "category",
    #                     "categories": [
    #                         "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Assoc-acdm",
    #                         "Assoc-voc", "Bachelors", "Doctorate", "HS-grad", "Masters", "Preschool",
    #                         "Prof-school", "Some-college"
    #                     ]
    #                 },
    #                 "fnlwgt": {"type": "int", "range": [0, 2000000]},
    #                 "hours-per-week": {"type": "int", "range": [0, 100]},
    #                 "marital-status": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Divorced", "Married-AF-spouse", "Married-civ-spouse",
    #                         "Married-spouse-absent", "Never-married", "Separated",
    #                         "Widowed"
    #                     ]
    #                 },
    #                 "native-country": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
    #                         "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
    #                         "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
    #                         "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico",
    #                         "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland",
    #                         "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
    #                         "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
    #                     ]
    #                 },
    #                 "occupation": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
    #                         "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service",
    #                         "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support",
    #                         "Transport-moving"
    #                     ]
    #                 },
    #                 "race": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"
    #                     ]
    #                 },
    #                 "relationship": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried",
    #                         "Wife"
    #                     ]
    #                 },
    #                 "sex": {"type": "category", "categories": ["Female", "Male"]},
    #                 "workclass": {
    #                     "type": "category",
    #                     "categories": [
    #                         "Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc",
    #                         "Self-emp-not-inc", "State-gov", "Without-pay"
    #                     ]
    #                 }
    #             },
    #             "y": {"name": "salary>50K", "type": "bool"}
    #         }
            
    #         Step 1 (this round): Generate multiple queries to answer the user's question. Use the following format to generate queries:
    #         <<age: [value] capital-gain: [value] capital-loss: [value] education-num: [value] education: [value] fnlwgt: [value] hours-per-week: [value] marital-status: [value] native-country: [value] occupation: [value] race: [value] relationship: [value} sex: [value] workclass: [value]>>
            
    #         Step 2 (this round): After generating the queries, explicitly state that you are waiting for results:
    #         "I have generated the above queries and am waiting for the actual results in the next round of dialogue. I will not make any conclusions or assumptions until I receive the results."
            
    #         Step 3 (next round): In the next round of dialogue, I will provide the results for your queries. Only then can you analyze the data and answer the question.
            
    #         Important notes:
    #         - In this round of dialogue, do not assume, generate, or infer any results.
    #         - Ensure your queries cover a sufficient sample to answer the question. (Typically more than 10 queries)
    #         - Do not provide any information or conclusions other than generating queries and the waiting statement.
    #         - Use <<>> symbols to enclose each query
    #         - Provide specific values for ALL fields. Do not use "Any" or ranges or skip any term.
    #         - Do not use ... or any other symbols to indicate continuation or omission of queries. Generate sufficient queries to cover the data.
            
    #         Now, please address the user's question: Which country has the highest average salary?
    #     '''
    
    msg_text = input(f"User {i}: ")
    messages.append({"role": "user", "content": msg_text})

    start_time = datetime.now()
    tokens = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    attn_mask = (tokens != tokenizer.pad_token_id).long()
    # print(f"Tokens: {tokenizer.decode(tokens.squeeze(0).tolist())}")

    generated_ids = model.generate(tokens, attention_mask=attn_mask, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1,
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    inference_time = datetime.now() - start_time

    #
    # perplexity = Perplexity(max_length=max_new_tokens, stride=max_new_tokens // 2, device="cuda")
    # ppl = perplexity.compute(model, generated_ids)
    # print(f"Perplexity: {ppl}")

    result = tokenizer.decode(generated_ids[0][tokens.size(1):].tolist())
    messages.append({"role": "assistant", "content": result})
    print(f"Response {i} (Time: {inference_time.total_seconds()}s): {result}")
    i += 1
