import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=256)

def load_model_and_inference_with_group(model_dir, inference_data_path, device, batch_size=32, output_file="results.json"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    inference_dataset = load_dataset('json', data_files={'inference': inference_data_path})['inference']
    inference_dataset = inference_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    inference_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'group', 'sentence'])
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    logits = defaultdict(list)
    results = []

    with torch.no_grad():
        for batch in tqdm(inference_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groups = batch['group']

            outputs = model(input_ids, attention_mask=attention_mask)
            for logit, group, sentence in zip(outputs.logits[:, 1].cpu().numpy(), groups, batch['sentence']):
                bias = "biased" if logit > 0 else "not biased"
                logit_rounded = round(float(logit), 4)
                logits[group].append(logit_rounded)
                results.append({
                    "group": group,
                    "sentence": sentence,
                    "logit": logit_rounded,
                    "bias": bias
                })

    group_averages = {group: round(float(sum(values)) / len(values), 4) for group, values in logits.items()}

    def convert_to_serializable(obj):
        if isinstance(obj, (float, np.float32)):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "group_averages": group_averages
        }, f, ensure_ascii=False, indent=4, default=convert_to_serializable)

    print(f"Results saved to {output_file}")
    return results, group_averages


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--inference_data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="results.json")

    args = parser.parse_args()

    results, group_averages = load_model_and_inference_with_group(
        args.model_dir, args.inference_data_path, args.device, args.batch_size, args.output_file
    )
    print("Group averages:", group_averages)
