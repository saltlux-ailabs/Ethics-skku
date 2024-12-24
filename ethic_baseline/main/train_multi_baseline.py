
import torch
import json
from torch.utils.data import DataLoader
from torch import nn
import os
from sklearn.metrics import f1_score, accuracy_score
import random
import numpy as np
# import pandas as pd
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
    default=23
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
parser.add_argument(
    "--mode",
    type=str,
    required=True,
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
            logits=outputs.logits

            probs=torch.sigmoid(logits)
            preds=(probs>0.5).float()

            all_predictions.append(preds.cpu().numpy().astype(np.int32))
            all_labels.append(labels.cpu().numpy().astype(np.int32))
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    f1_per_class = list(f1_score(all_labels, all_predictions, average=None))
    f1_micro=f1_score(all_labels,all_predictions,average='micro')
    f1_macro=f1_score(all_labels,all_predictions,average='macro')
    f1_weighted=f1_score(all_labels,all_predictions,average='weighted')
    total_accuracy=accuracy_score(all_labels,all_predictions)
    print(total_accuracy)

    accuracy_per_class = []
    for i in range(all_labels.shape[1]):
        # if i==0:
        #     print(all_predictions[:i])
        score=accuracy_score(all_labels[:,i],all_predictions[:,i])
        accuracy_per_class.append(score)
    result={
        'f1_per_class': f1_per_class,
        "accuracy_per_class:": accuracy_per_class,
        "f1_micro:": f1_micro,
        "f1_macro:": f1_macro,
        "f1_weighted":f1_weighted,
        "total_accuracy:": total_accuracy,
    }
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

args.num_labels= 3 if args.mode=='problems' else 8 # args.num_labels= 4 if args.mode=='problems' else 8
model=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=args.num_labels, problem_type="multi_label_classification")

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn=nn.BCEWithLogitsLoss()

num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
print(f'(with linear decay scheduler) Total training step : {num_training_steps}, warmup step : {warmup_ratio*num_training_steps}')
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio*num_training_steps,num_training_steps=num_training_steps)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model.to(device)
best_f1=-1
best_epch=-1
best_acc=-1
total_steps=0
# result = evaluate_model(model, test_dataloader)
# print(result)
# writer.add_scalar('f1', result['f1'], 0)
# writer.add_scalar('acc', result['accuracy'], 0)

save_dir = "/nlp_data/yumin/ethic_baseline/models/multi_baseline"
best_f1_score=0

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

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())
        logits = outputs.logits

        loss=loss_fn(logits, labels.float())
        writer.add_scalar("Loss/train", loss.item(), total_steps)
        total_steps+=1

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        scheduler.step()
    result = evaluate_model(model, test_dataloader)

    # print(result["total_accuracy"])

    f1_weighted = result["f1_weighted"]
    if f1_weighted > best_f1_score:
        best_f1_score = f1_weighted
        best_epch = epoch + 1
        best_result = {
            "epoch": best_epch,
            "f1_weighted": best_f1_score
        }
        print(f1_weighted)
        print(f"New best model found at epoch {epoch + 1}, saving model...")

        # model.save_pretrained(save_dir)
        # tokenizer.save_pretrained(save_dir)

        # with open(os.path.join(save_dir, "training_args.json"), "w") as f:
        #     json.dump(vars(args), f, indent=4)
        # torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

writer.close()
print('@@@@@ Training completed')