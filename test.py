import os
import nvitop
import time
import sys
import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
from tqdm import tqdm
from utils.prompter import Prompter
from sklearn.metrics import recall_score, precision_score, accuracy_score


def compute(pred, label):
    acc = accuracy_score(label, pred)
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    f1 = (2 * precision * recall) / (precision + recall)
    return acc, precision, recall, f1

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

just_instruction = """Did the changes cause any issues with consistency in the {}?
```changes
{}
```
```{}
{}
```
"""

post_instruction = """Is the given code consistent with the corresponding {}?
```code
{}
```
```{}
{}
```
"""

def main(
    base_model: str = '/nvme1n1/LLM/CodeLlama-13b-hf',
    lora_weights: str = "",
    prompt_template: str = "llama",
    out_path: str = "Data/TestResult.xlsx",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if lora_weights:
            lora_model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True
        )
        if lora_weights:
            lora_model = PeftModel.from_pretrained(
                model,
                lora_weights,
            )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    if lora_weights:
        lora_model.config.pad_token_id = 0

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    @torch.inference_mode()
    def evaluate(
        model,
        instruction,
        input=None,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        res = prompter.get_response(output)
        return res

    results = {"old_comment_raw":[], 'new_code_raw':[], 'new_comment_raw':[], "label":[], "output":[], "flag":[]}
    flags = []
    labels = []

    for class_ in ["Summary", "Param", "Return"]:
        with open(f"Data/{class_}/test.json") as f:
            data = json.load(f)

        for example in tqdm(data):
            try:
                res = evaluate(model, post_instruction.format(class_.lower(),example['new_code_raw'],class_.lower(),example['old_comment_raw']))
                results["old_comment_raw"].append(example["old_comment_raw"])
                results["new_code_raw"].append(example["new_code_raw"])
                results["new_comment_raw"].append(example["new_comment_raw"])
                results["label"].append(example["label"])
                judge = 1 if "inconsisten" in res else 0
                flags.append(judge)
                labels.append(example["label"])
                results["flag"].append(judge)
                results["output"].append(res)
            except Exception as e:
                print(e)
                pass

    acc, precision, recall, f1 = compute(flags, labels)

    df = pd.DataFrame(results)
    df.to_excel(out_path, index=False)
    with open("result.txt", "a+") as f:
        f.write(f"base_model: {base_model}, lora_weights:{lora_weights}, acc: {acc:.4f}, precision:{precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n")


if __name__ == "__main__":
    fire.Fire(main)
