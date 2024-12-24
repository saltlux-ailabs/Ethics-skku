import torch
import json
from torch.utils.data import DataLoader
from torch import nn
import os
from sklearn.metrics import f1_score, accuracy_score
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

from transformers.utils import logging
logging.set_verbosity_error() 

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
    default=10,
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
parser.add_argument(
    "--loss_lambda",
    type=float,
    default=10.0,
)
parser.add_argument(
    "--loss_alpha",
    type=float,
    default=0.7,
)
parser.add_argument(
    "--loss_temp",
    type=float,
    default=2.0,
)
parser.add_argument(
    "--find_hyper",
    action='store_true',
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

    accuracy_per_class = []
    for i in range(all_labels.shape[1]):
        # if i==0:
        #     print(all_predictions[:i])
        score=accuracy_score(all_labels[:,i],all_predictions[:,i])
        accuracy_per_class.append(score)
    result={
        'f1_per_class': f1_per_class,
        "accuracy_per_class": accuracy_per_class,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted":f1_weighted,
        "total_accuracy": total_accuracy,
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
    preprocess_data = tokenizer(examples['sentence'], truncation=True, padding='max_length', max_length=256, add_special_tokens=True)
    entity_mask = np.zeros((len(examples['sentence']), 256))

    for i, data_i in enumerate(examples['NER_results']):
        for j, t_i in enumerate(data_i):
            if t_i != "O":
                entity_mask[i][j] = 1

    preprocess_data['entity_mask'] = entity_mask

    return preprocess_data

# 데이터셋 전처리
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'entity_mask'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

args.num_labels= 3 if args.mode=='problems' else 8 # args.num_labels= 4 if args.mode=='problems' else 8
model=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=args.num_labels, problem_type="multi_label_classification")

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

class DistillationBCELoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(DistillationBCELoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()

    def forward(self, student_logits, teacher_logits, target):
        # Convert logits to probabilities using sigmoid
        teacher_prob = torch.sigmoid(teacher_logits / self.temperature)
        student_prob = torch.sigmoid(student_logits)

        # Distillation loss (BCE between teacher and student probabilities)
        distillation_loss = self.bce_loss(student_prob, teacher_prob)

        # Standard BCE loss (between student and ground truth target)
        standard_bce_loss = self.bce_loss(student_prob, target)

        # Combine losses (weighted sum of distillation loss and standard loss)
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * standard_bce_loss

        return total_loss

def get_valid_attention_probs(attention_score, attention_mask, ner_mask):
    all_attentions = torch.stack(attention_score)

    n_layer, bs, n_head, n_token, _ = all_attentions.shape

    all_attentions = all_attentions.permute(0, 2, 3, 1, 4).contiguous()
    all_attentions = all_attentions.view(n_layer, n_head, n_token, -1)

    attention_mask = (attention_mask.flatten() - ner_mask.flatten()).bool()

    all_attentions[:, :, :, ~attention_mask] = 0.0 # mask unvalid attention scores (attention for unvalid token)
    all_attentions = all_attentions.view(n_layer, n_head, n_token, bs, -1)
    all_attentions = all_attentions.permute(0, 1, 3, 2, 4).contiguous().view(n_layer, n_head, n_token * bs, -1)

    no_mask_attentions = all_attentions[:, :, attention_mask, : ] # remain valid attention scores (attention from unvalid token)

    return no_mask_attentions, attention_mask

bce_loss_fn = nn.BCEWithLogitsLoss()
bce_distill_loss_fn = DistillationBCELoss(temperature=args.loss_temp, alpha=args.loss_alpha)
kl_loss_fn = nn.KLDivLoss(reduction='none')

num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
# print(f'(with linear decay scheduler) Total training step : {num_training_steps}, warmup step : {warmup_ratio*num_training_steps}')
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio*num_training_steps,num_training_steps=num_training_steps)
device='cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

model.to(device)
best_f1=-1
best_acc=-1
best_epoch=-1
total_steps=0
c_debias = args.loss_lambda # kl div ~ 0.1

# result = evaluate_model(model, test_dataloader)
# print(result)
# writer.add_scalar('f1', result['f1'], 0)
# writer.add_scalar('acc', result['accuracy'], 0)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        # 입력 및 출력 설정
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        entity_mask = batch['entity_mask']
        masked_input_ids = batch['input_ids']
        masked_input_ids[batch['entity_mask'].bool()] = 4 # [MASK] token
        
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        masked_input_ids=masked_input_ids.to(device)
        labels=labels.to(device)
        entity_mask = entity_mask.to(device)

        bs, n_token = input_ids.shape
        
        # 모델 출력 및 손실 계산
        # original input
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float(), output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions

        # Original Loss
        bce_loss = bce_loss_fn(logits, labels.float())

        teacher_logits = logits.clone().detach()

        # print(model.device)
        # print(masked_input_ids.device)
        # print(masked_input_ids[:, :20])
        # print(attention_mask.device)
        # print(labels.device)

        masked_outputs = model(masked_input_ids, attention_mask=attention_mask, labels=labels.float(), output_attentions=True)
        masked_logits = masked_outputs.logits
        masked_attentions = masked_outputs.attentions

        # Knowledge Distillation for Masked Input
        masked_bce_loss = bce_distill_loss_fn(masked_logits, teacher_logits, labels.float())

        # Knowledge Distillation for Debiasing
        valid_masked_attentions, total_mask = get_valid_attention_probs(masked_attentions, attention_mask, entity_mask)
        valid_masked_attentions = valid_masked_attentions.clone().detach()

        valid_attentions, _ = get_valid_attention_probs(attentions, attention_mask, entity_mask)
        log_valid_attentions = torch.log(valid_attentions + 1e-9) # avoid log(0)

        attention_kl = kl_loss_fn(log_valid_attentions, valid_masked_attentions) # n_layer * n_attention_head * valid attention distribution * n_token

        index_data = torch.arange(bs).repeat(n_token, 1).permute(1, 0).contiguous().flatten().to(attention_kl.device)
        valid_index_data = index_data[total_mask]
        denominator = torch.bincount(valid_index_data, minlength=bs).to(attention_kl.device)

        debias_loss = attention_kl.sum(-1) / denominator[valid_index_data]

        # mean of all attention_kl
        debias_loss = torch.mean(debias_loss)

        loss = bce_loss + masked_bce_loss + c_debias * debias_loss

        writer.add_scalar("BCE_Loss/train", bce_loss.item(), total_steps) 
        writer.add_scalar("KD_Loss/train", masked_bce_loss.item(), total_steps) 
        writer.add_scalar("DEBIAS_Loss/train", debias_loss.item(), total_steps)        
        writer.add_scalar("Loss/train", loss.item(), total_steps)
        total_steps+=1

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        scheduler.step()

    result = evaluate_model(model, test_dataloader)

    if best_acc < result['total_accuracy']:
        best_f1 = result['f1_micro']
        best_epoch = epoch
        best_acc = result['total_accuracy']
#    print(f'$#$#$#RESULT {model_id.split("/")[-1]} Epoch {epoch+1} finished')
#    print(f'{model_id} epoch {epoch+1} result')
#    print(result)
    # writer.add_scalar('f1', result['f1'], epoch)
    # writer.add_scalar('acc', result['accuracy'], epoch)

if args.find_hyper:
    name = ",".join([args.hyper_param, str(args.batch_size), str(args.loss_alpha), str(args.loss_temp), str(args.loss_lambda)])

    with open('./find_hyper.json', 'r') as f:
        logs = json.load(f)

    logs[name] = {'best_epoch': best_epoch, 'best_f1': best_f1, 'best_acc': best_acc}

    with open('./find_hyper.json', 'w') as f:
        json.dump(logs, f, ensure_ascii=False, indent = 4)
        
writer.close()
# print('@@@@@ Training completed')
