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
import ear
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AdamW,
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


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
parser.add_argument("--reg_strength", type=float, default=0.01)
parser.add_argument("--save_dir", type=str, default="saved_models")

args=parser.parse_args()
save_dir = args.save_dir

run_name = f"reg_strength_{args.reg_strength}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"{args.logdir}/{run_name}")
seed_everything(args.seed)

model_id=args.model
lr=float(args.hyper_param.split(',')[0])
warmup_ratio=float(args.hyper_param.split(',')[1])
weight_decay=float(args.hyper_param.split(',')[2])
batch_size=args.batch_size

def compute_negative_entropy(
    inputs: tuple, attention_mask: torch.torch, return_values: bool = False
):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS
    """
    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    if return_values:
        return final_entropy, neg_entropies
    else:
        return final_entropy

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
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

args.num_labels= 3 if args.mode=='problems' else 8 # args.num_labels= 4 if args.mode=='problems' else 8
model=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=args.num_labels, problem_type="multi_label_classification")

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

bce_loss_fn = nn.BCEWithLogitsLoss()

reg_strength = args.reg_strength

num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
print(f'(with linear decay scheduler) Total training step : {num_training_steps}, warmup step : {warmup_ratio*num_training_steps}')
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_ratio*num_training_steps,num_training_steps=num_training_steps)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model.to(device)
best_f1_score=-1
best_epch=-1
total_steps=0

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
        # original input
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float(), output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions

        # Original Loss
        bce_loss = bce_loss_fn(logits, labels.float())

        # neg_entropy = ear.compute_negative_entropy(
        #     inputs=outputs.attentions,
        #     attention_mask=attention_mask
        # )
        neg_entropy = compute_negative_entropy(
            inputs=outputs.attentions,
            attention_mask=attention_mask
        )
        reg_loss = reg_strength * neg_entropy
        reg_loss = torch.mean(reg_loss)

        loss = bce_loss + reg_loss

        writer.add_scalar("BCE_Loss/train", bce_loss.item(), total_steps) 
        writer.add_scalar("reg_loss/train", reg_loss.item(), total_steps)         
        writer.add_scalar("Loss/train", loss.item(), total_steps)
        total_steps+=1

        # 역전파 및 옵티마이저 스텝
        loss.backward()
        optimizer.step()
        scheduler.step()

    result = evaluate_model(model, test_dataloader)
    f1_weighted = result["f1_weighted"]
    print(f'Epoch {epoch + 1} F1 Weighted Score: {f1_weighted}')

    writer.add_scalar("F1_Score/weighted", f1_weighted, epoch + 1)

    if f1_weighted > best_f1_score:
        best_f1_score = f1_weighted
        best_epch = epoch + 1
        best_result = {
            "epoch": best_epch,
            "f1_weighted": best_f1_score,
            "hyper_parameters": {
                "reg_strength": reg_strength,
                "learning_rate": lr,
                "batch_size": batch_size
            }
        }
        print(f"New best model found at epoch {epoch + 1}, saving model...")

        # model.save_pretrained(save_dir)
        # tokenizer.save_pretrained(save_dir)
        # torch.save(args, os.path.join(save_dir, "training_args.bin"))
        # torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

experiment_result = {
    "f1_weighted": best_f1_score,
    "epoch": best_epch,
    "hyper_parameters": {
        "reg_strength": reg_strength,
        "learning_rate": lr,
        "batch_size": batch_size
    }
}

results_path = os.path.join(args.logdir, "all_results.txt")
with open(results_path, "a") as f:
    f.write(json.dumps(experiment_result, indent=4) + "\n")

writer.close()
print('Training completed')