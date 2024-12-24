
import torch
import json
from torch.utils.data import DataLoader
import os
import evaluate
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AdamW,
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--model",
    type=str,
)
parser.add_argument(
    "--hyper_param", # "lr,warmup_ratio,weight_decay"
    type=str,
)
parser.add_argument(
    "--batch_size", # "lr,warmup_ratio,weight_decay"
    type=int,
    default=32
)
parser.add_argument( ###
    '--num_labels',
    type=int,
    default=2,
)
parser.add_argument(
    "--train_data",
    type=str,
    # default="/nlp_data/yumin/kosbi/baseline/preprocessed_data_kor/kosbi_v2_train.json",
)
parser.add_argument(
    "--valid_data",
    type=str,
    # default="/nlp_data/yumin/kosbi/baseline/preprocessed_data_kor/kosbi_v2_valid.json",
)
parser.add_argument(
    "--test_data",
    type=str,
    # default="/nlp_data/yumin/kosbi/baseline/preprocessed_data_kor/kosbi_v2_test.json",
)

parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
)
parser.add_argument(
    "--logdir",
    type=str,
    default='log',
)
# args 정보 출력하기

args=parser.parse_args()

writer = SummaryWriter(args.logdir)
seed_everything(args.seed)

model_id=args.model
lr=float(args.hyper_param.split(',')[0])
warmup_ratio=float(args.hyper_param.split(',')[1])
weight_decay=float(args.hyper_param.split(',')[2])
batch_size=args.batch_size

def evaluate_model(model, dataloader):
    model.eval()
    all_labels=[]
    all_predictions=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            labels=labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            _, predicted = torch.max(outputs.logits, dim=1)
            all_labels.extend(labels)
            all_predictions.extend(predicted)
    metrics=evaluate.combine(
        ['accuracy','f1']
    )
    result=metrics.compute(predictions=all_predictions, references=all_labels)
    return result

tokenizer=AutoTokenizer.from_pretrained(model_id)

# JSON 데이터셋 로드
data_files={
    'train':args.train_data,
    'test':args.test_data,
    'valid': args.valid_data
}
dataset = load_dataset('json', data_files=data_files)#, streaming=True)

train_dataset=dataset['train']
test_dataset=dataset['test']

print(next(iter(train_dataset)))
def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=256, add_special_tokens=True)

# 데이터셋 전처리
train_dataset = train_dataset.map(preprocess_function,batched=True)
test_dataset = test_dataset.map(preprocess_function,batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=args.num_labels)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
print(f'(with linear decay scheduler) Total training step : {num_training_steps}, warmup step : {warmup_ratio*num_training_steps}')
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio*num_training_steps,num_training_steps=num_training_steps)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model.to(device)
best_f1=-1
best_epch=-1
total_steps=0
# result = evaluate_model(model, test_dataloader)
# print(result)
# writer.add_scalar('f1', result['f1'], 0)
# writer.add_scalar('acc', result['accuracy'], 0)

best_model = None

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        # 입력 및 출력 설정
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        labels=labels.to(device)
        
        # 모델 출력 및 손실 계산
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        writer.add_scalar("Loss/train", loss.item(), total_steps)
        total_steps+=1

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f'$#$#$#RESULT {model_id.split("/")[-1]} Epoch {epoch+1} finished')
    print(f'{model_id} epoch {epoch+1} result')
    result = evaluate_model(model, test_dataloader)
    print(result)

    # if best_model:
    #     if best_f1 < result['f1']:
    #         torch.save(model.state_dict(), f'../models/ethic_bin_baseline_{model_id.replace("/","-")}.pt')
    # else:
    #     torch.save(model.state_dict(), f'../models/ethic_bin_baseline_{model_id.replace("/","-")}.pt')

    save_dir = "/nlp_data/yumin/ethic_baseline/models/binary_baseline"
    if best_f1 < result['f1']:
        best_f1 = result['f1']
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    writer.add_scalar('f1', result['f1'], epoch)
    writer.add_scalar('acc', result['accuracy'], epoch)
writer.close()
print('@@@@@ Training completed')