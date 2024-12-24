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
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)  # 라벨 개수를 3으로 설정
    model.to(device)
    model.eval()

    # 데이터셋 로드 및 전처리
    inference_dataset = load_dataset('json', data_files={'inference': inference_data_path})['inference']
    inference_dataset = inference_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # 필요한 필드 포함
    inference_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'group', 'sentence'])
    inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    logits = defaultdict(list)
    logits_raw = defaultdict(list)
    results = []

    with torch.no_grad():
        for batch in tqdm(inference_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            groups = batch['group']

            # 모델 예측
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)  # Sigmoid로 확률 값 계산

            for logitt, prob, group, sentence in zip(outputs.logits.cpu().numpy(), probs.cpu().numpy(), groups, batch['sentence']):
                labels = (prob > 0.5).astype(int)  # 0.5 기준으로 멀티 라벨 예측
                probs_rounded = [round(float(p), 4) for p in prob]  # 소숫점 4자리로 반올림
                logit_rounded = [round(float(p), 4) for p in logitt]
                logits[group].append(probs_rounded)
                logits_raw[group].append(logit_rounded)
                results.append({
                    "group": group,
                    "sentence": sentence,
                    "probs": probs_rounded,
                    "logit": logit_rounded,
                    "labels": labels.tolist()  # 예측된 라벨 리스트
                })

    # 그룹별 평균 계산 (라벨별 평균 확률)
    group_averages = {
        group: [round(float(sum(prob)) / len(prob), 4) for prob in zip(*values)]
        for group, values in logits.items()
    }

    logit_averages = {
    group: [round(float(sum(logit)) / len(logit), 4) for logit in zip(*values)]
    for group, values in logits_raw.items()
}
    print("Group logits averages:", logit_averages)

    # 결과 저장
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
