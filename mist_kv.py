import os
import subprocess
import sys
import time

os.environ["PATH"] += "/data/home1/shishirm/.local/lib/python3.8/site-packages" + os.pathsep

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from copy import deepcopy
import json
from typing import List, Optional, Tuple, Type, TypeVar
from tqdm import tqdm
from pydantic.dataclasses import dataclass

device = "cuda" # the device to load the model onto
attn_implementation = "sdpa"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, attn_implementation=attn_implementation, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The dataclass code is taken from the author's github repository.
T = TypeVar("T")

@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))
    
def get_kv_retrieval_prompt(data: List[Tuple[str, str]], key: str, query_aware_contextualization: bool):
    if query_aware_contextualization:
        with open("./prompting/kv_retrieval_with_query_aware_contextualization.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    else:
        with open("./prompting/kv_retrieval.prompt") as f:
            prompt_template = f.read().rstrip("\n")
    
    # Format the KV data into a string
    formatted_kv_records = ""
    for index, record in enumerate(data):
        start_character = "{" if index == 0 else " "
        data_string = f'"{record[0]}": "{record[1]}"'
        end_character = ",\n" if index != len(data) - 1 else "}"
        formatted_kv_records += start_character + data_string + end_character
        
    return prompt_template.format(formatted_kv_records=formatted_kv_records, key=key)


def get_responses(inp : str, out : str, correct_index : int, query_aware_contextualization: bool):
    prompts = []
    responses = []
    correct_responses = []
    durations = []
    
    with open(inp) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            
            # Getting the kv records, correct key & value
            ordered_kv_records = deepcopy(input_example["ordered_kv_records"])
            key = input_example["key"]
            value = input_example["value"]

            # Making correct_index to have the correct key-value
            original_kv_index = ordered_kv_records.index([key, value])
            original_kv = ordered_kv_records.pop(original_kv_index)
            ordered_kv_records.insert(correct_index, original_kv)

            kv_prompt = get_kv_retrieval_prompt(
                data=ordered_kv_records, key=key, query_aware_contextualization=query_aware_contextualization
            )
            
            prompts.append(kv_prompt)
            correct_responses.append(value)
            
            encodeds = tokenizer([kv_prompt], return_tensors="pt")
            model_inputs = {key: value.to("cuda") for key, value in encodeds.items()}
            # model.to(device)
            start = time.process_time()
            generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100, do_sample=False)
            decoded = tokenizer.batch_decode(generated_ids)
            durations.append(time.process_time() - start)
            
            responses.append(decoded[0])
            
            
    with open(out, "w") as f:
        for prompt, response, correct_answer, duration in zip(prompts, responses, correct_responses, durations):
            output = {}

            output["model_prompt"] = prompt
            output["model_answer"] = response
            output["model"] = model_name
            output["correct_answer"] = correct_answer
            output["time"] = duration

            f.write(json.dumps(output) + "\n")


input_paths = [
    "./kv_retrieval_data/kv-retrieval-75_keys.jsonl",
    "./kv_retrieval_data/kv-retrieval-140_keys.jsonl",
    "./kv_retrieval_data/kv-retrieval-300_keys.jsonl",
    ]
output_paths = [
    [ "./mist_kv/mist_7b_sdpa_kv_75_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_75_key_at_24_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_75_key_at_49_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_75_key_at_74_responses.jsonl"], 
    [ "./mist_kv/mist_7b_sdpa_kv_140_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_140_key_at_34_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_140_key_at_69_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_140_key_at_104_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_140_key_at_139_responses.jsonl"],
    [ "./mist_kv/mist_7b_sdpa_kv_300_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_49_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_99_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_149_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_199_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_249_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_300_key_at_299_responses.jsonl"]
    ]
                
correct_indices = [[0, 24, 49, 74], [0, 34, 69, 104, 139], [0, 49, 99, 149, 199, 249, 299]]

for i in range(len(input_paths)):
    for j in range(len(output_paths[i])):
        # print(input_paths[i], output_paths[i][j], correct_indices[i][j])
        get_responses(input_paths[i], output_paths[i][j], correct_indices[i][j], False)

    
output_paths = [
    [ "./mist_kv/mist_7b_sdpa_kv_QAC_75_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_75_key_at_24_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_75_key_at_49_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_75_key_at_74_responses.jsonl"], 
    [ "./mist_kv/mist_7b_sdpa_kv_QAC_140_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_140_key_at_34_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_140_key_at_69_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_140_key_at_104_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_140_key_at_139_responses.jsonl"],
    [ "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_0_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_49_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_99_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_149_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_199_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_249_responses.jsonl",
      "./mist_kv/mist_7b_sdpa_kv_QAC_300_key_at_299_responses.jsonl"]
    ]


for i in range(len(input_paths)):
    for j in range(len(output_paths[i])):
        # print(input_paths[i], output_paths[i][j], correct_indices[i][j])
        get_responses(input_paths[i], output_paths[i][j], correct_indices[i][j], True)