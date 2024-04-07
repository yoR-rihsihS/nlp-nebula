import os
import re
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install("ninja")

# os.environ["PATH"] += os.pathsep + "" # This was to temporarily add ninja library path from another python installation


# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries otherwise '0'

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from copy import deepcopy
import json
from typing import List, Optional, Tuple, Type, TypeVar
from tqdm import tqdm
from pydantic.dataclasses import dataclass

# download models: https://huggingface.co/BlinkDL
model_name = "RWKV-4-Raven-14B-v12-ctx8192"
model = RWKV(model = './models/RWKV-4-Raven-14B-v12-ctx8192.pth', strategy = 'cuda fp16')
pipeline = PIPELINE(model, "./model/20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# use pipeline = PIPELINE(model, "rwkv_vocab_v20230424") for rwkv "world" models

# For Greedy Decoding
args = PIPELINE_ARGS(temperature = 0.0, top_p = 1.0, top_k = 0, # top_k = 0 then ignore
                     alpha_frequency = 0.25,
                     alpha_presence = 0.25,
                     alpha_decay = 0.996, # gradually decay the penalty
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0], # stop generation whenever you see any token here
                     chunk_len = 1024) # split input into chunks to save VRAM (shorter -> slower)


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
    


def get_qa_prompt(question: str, documents: List[Document], query_aware_contextualization: bool):
    if query_aware_contextualization:
        prompt_path = "./prompting/qa_with_query_aware_contextualization.prompt"
    else:
        prompt_path = "./prompting/qa.prompt"
        
    with open(prompt_path) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_oracle_responses(inp : str, out : str, query_aware_contextualization: bool):
    prompts = []
    responses = []
    correct_responses = []
    
    with open(inp) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            
            # Getting question and correct answer
            question = input_example["question"]
            correct_answer = input_example["answers"]
            
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            
            qa_prompt = get_qa_prompt(question, documents, query_aware_contextualization)
            
            prompts.append(qa_prompt)
            correct_responses.append(correct_answer)

            answer = pipeline.generate(qa_prompt, token_count=100, args=args)
            responses.append(answer)
             
            
    with open(out, "w") as f:
        for prompt, response, correct_answer in zip(prompts, responses, correct_responses):
            output = {}

            output["model_prompt"] = prompt
            output["model_answer"] = response
            output["model"] = model_name
            output["correct_answer"] = correct_answer

            f.write(json.dumps(output) + "\n")


def get_multi_doc_qa_responses(inp : str, out : str, query_aware_contextualization: bool):
    prompts = []
    responses = []
    correct_responses = []
    
    with open(inp) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            
            # Getting question and correct answer
            question = input_example["question"]
            correct_answer = input_example["answers"]
            
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            
            qa_prompt = get_qa_prompt(question, documents, query_aware_contextualization)
            
            prompts.append(qa_prompt)
            correct_responses.append(correct_answer)
            
            answer = pipeline.generate(qa_prompt, token_count=100, args=args)
            responses.append(answer)
             
            
    with open(out, "w") as f:
        for prompt, response, correct_answer in zip(prompts, responses, correct_responses):
            output = {}

            output["model_prompt"] = prompt
            output["model_answer"] = response
            output["model"] = model_name
            output["correct_answer"] = correct_answer

            f.write(json.dumps(output) + "\n")



input_path = "./qa_data/nq-open-oracle.json"
output_path = "./responses/rwkv_raven_14b_qa/raven_14b_oracle_responses.jsonl"

get_oracle_responses(input_path, output_path, False)

output_path = "./responses/rwkv_raven_14b_qa/raven_14b_oracle_QAC_responses.jsonl"

get_oracle_responses(input_path, output_path, True)


def get_closedbook_qa_prompt(question: str):
    with open("./prompting/closedbook_qa.prompt") as f:
        prompt_template = f.read().rstrip("\n")

    return prompt_template.format(question=question)


def get_closedbook_responses(inp, out):
    prompts = []
    responses = []
    correct_responses = []
    
    with open(inp) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            
            # Getting question and correct answer
            question = input_example["question"]
            correct_answer = input_example["answers"]
            
            qa_closedbook_prompt = get_closedbook_qa_prompt(question)
            
            prompts.append(qa_closedbook_prompt)
            correct_responses.append(correct_answer)
            
            answer = pipeline.generate(qa_closedbook_prompt, token_count=100, args=args)
            responses.append(answer)
             
            
    with open(out, "w") as f:
        for prompt, response, correct_answer in zip(prompts, responses, correct_responses):
            output = {}

            output["model_prompt"] = prompt
            output["model_answer"] = response
            output["model"] = model_name
            output["correct_answer"] = correct_answer

            f.write(json.dumps(output) + "\n")


input_path = "./qa_data/nq-open-oracle.json"
output_path = "./responses/rwkv_raven_14b_qa/raven_14b_closedbook_responses.jsonl"

get_closedbook_responses(input_path, output_path)



input_paths = [
    "./qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl",
    "./qa_data/10_total_documents/nq-open-10_total_documents_gold_at_4.jsonl",
    "./qa_data/10_total_documents/nq-open-10_total_documents_gold_at_9.jsonl"]

output_paths = [
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_0_responses.jsonl",
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_4_responses.jsonl",
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_9_responses.jsonl"]

for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], False)

output_paths = [
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_0_QAC_responses.jsonl",
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_4_QAC_responses.jsonl",
    "./responses/rwkv_raven_14b_qa/raven_14b_10_doc_at_9_QAC_responses.jsonl"]

for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], True)


input_paths = [
    "./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl",
    "./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_4.jsonl",
    "./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl",
    "./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_14.jsonl",
    "./qa_data/20_total_documents/nq-open-20_total_documents_gold_at_19.jsonl"]

output_paths = ["./responses/rwkv_raven_14b_qa/raven_14b_20_doc_at_0_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_doc_at_4_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_doc_at_9_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_doc_at_14_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_doc_at_19_responses.jsonl"]


for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], False)


output_paths = ["./responses/rwkv_raven_14b_qa/raven_14b_20_QAC_doc_at_0_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_QAC_doc_at_4_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_QAC_doc_at_9_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_QAC_doc_at_14_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_20_QAC_doc_at_19_responses.jsonl"]

for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], True)


input_paths = [
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_0.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_4.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_9.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_14.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_19.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_24.jsonl",
    "./qa_data/30_total_documents/nq-open-30_total_documents_gold_at_29.jsonl"]

output_paths = ["./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_0_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_4_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_9_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_14_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_19_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_24_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_29_responses.jsonl"]

for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], False)

output_paths = ["./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_0_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_4_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_9_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_14_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_19_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_24_QAC_responses.jsonl",
               "./responses/rwkv_raven_14b_qa/raven_14b_30_doc_at_29_QAC_responses.jsonl"]

for i in range(len(input_paths)):
    get_multi_doc_qa_responses(input_paths[i], output_paths[i], True)