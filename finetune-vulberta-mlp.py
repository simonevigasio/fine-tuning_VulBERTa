import torch 
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import sklearn

import numpy as np
import pandas as pd

import json

import time

import os

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_valid_enc = open('./datasets/diversevul/encodings/valid.json', 'r')
file_train_enc = open('./datasets/diversevul/encodings/train.json', 'r')

valid_enc = json.load(file_valid_enc)
train_enc = json.load(file_train_enc)

file_valid_data = open('./datasets/diversevul/data/valid.json', 'r')
file_train_data = open('./datasets/diversevul/data/train.json', 'r')

valid_data = json.load(file_valid_data)
train_data = json.load(file_train_data)

assert len(valid_data) == len(valid_enc['input_ids']) == len(valid_enc['attention_mask'])
assert len(train_data) == len(train_enc['input_ids']) == len(train_enc['attention_mask'])

def process_labels(dataset):
  labels = []
  for data in dataset:
    label = 0
    if 'pillar' in data:
      label = 1
    labels.append(label)
  return labels

valid_labels = process_labels(valid_data)
train_labels = process_labels(train_data)

n, p = 0, 0
for l in train_labels: 
  if l == 0: n += 1
  else: p += 1

print(p/(n+p))

class VulnerabilityDetectiondDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels
    assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) ==  len(self.labels)

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

valid = VulnerabilityDetectiondDataset(encodings=valid_enc, labels=valid_labels)
train = VulnerabilityDetectiondDataset(encodings=train_enc, labels=train_labels)

# valid_loader = DataLoader(valid, batch_size=8, shuffle=True)
# train_loader = DataLoader(train, batch_size=8, shuffle=True)

model = RobertaForSequenceClassification.from_pretrained('./models/vulberta-pretrained')
model.to(device)
modules = [model.roberta.embeddings, model.roberta.encoder.layer[:5]]
for module in modules: 
  for param in module.parameters():
    param.requires_grad = False

print(model)
print(model.num_parameters())

class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_labels)
class_weights = torch.FloatTensor([class_weights[0], class_weights[1]])
criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = criterion.to(device)

# ----------------------------------------------------------------- #

class MyTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs["logits"]
    loss = criterion(logits, labels)
    return (loss, outputs) if return_outputs else loss
  
training_args = TrainingArguments(
  output_dir='./models/vulberta-mlp-generic',
  overwrite_output_dir=False,
  per_device_train_batch_size=8,
  num_train_epochs=10,
  eval_strategy='epoch',
  save_strategy='epoch',
  save_total_limit=20, # save last 20 checkpoints
  seed=seed,
  learning_rate=3e-05,
  fp16=True, # 16-bit floating point 
  report_to=None,
  load_best_model_at_end =True
)

trainer = MyTrainer(
  model=model,
  args=training_args,
  train_dataset=train,
  eval_dataset=valid 
)

trainer.train()

# ----------------------------------------------------------------- #

# optimizer = Adam(model.parameters(), lr=0.0005)

# model_folder = './models/vulberta-mlp-cwe-664'
# if not os.path.exists(model_folder):
#   os.makedirs(model_folder)

# def get_class(x):
#   return(x.index(max(x)))

# print('training started...')

# epoch = 20

# for e in range(epoch):
#   timer = time.time()

#   run_acc, run_loss = 0, 0
#   model.train()

#   for batch in train_loader:
#     input_ids = batch['input_ids'].to(device)
#     attention_mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)

#     optimizer.zero_grad()
#     output = model(input_ids, attention_mask=attention_mask, labels=labels)

#     loss = output[0]
#     loss.backward()

#     optimizer.step()

#     probs = torch.nn.functional.softmax(output[1], dim=1)
#     probs = pd.Series(probs.tolist())
#     preds = probs.apply(get_class)
#     preds.reset_index(drop=True, inplace=True)
#     value_counts = pd.value_counts(preds == labels.tolist())

#     try:
#       acc = value_counts[1]/len(labels.tolist())
#     except:
#       if(value_counts.index[0]==False):
#         acc = 0
#       else:
#         acc = 1

#     run_acc += acc
#     run_loss += loss.item()

#   with torch.no_grad():
#     run_acc_val, run_loss_val = 0, 0
#     model.eval()

#     for batch in valid_loader:
#       input_ids_val = batch['input_ids'].to(device)
#       attention_mask_val = batch['attention_mask'].to(device)
#       labels_val = batch['labels'].to(device)

#       output_val = model(input_ids_val, attention_mask=attention_mask_val, labels=labels_val)
#       loss_val = output[0]

#       probs_val = torch.nn.functional.softmax(output_val[1], dim=1)
#       probs_val = pd.Series(probs_val.tolist())
#       preds_val = probs_val.apply(get_class)
#       preds_val.reset_index(drop=True, inplace=True)
#       value_counts_val = pd.value_counts(preds_val == labels_val.tolist())

#       try:
#         acc_val = value_counts_val[1]/len(labels_val.tolist())
#       except:
#         if(value_counts_val.index[0]==False):
#           acc_val = 0
#         else:
#           acc_val = 1

#       run_acc_val += acc_val
#       run_loss_val += loss_val.item()

#   print_out = 'Epoch %d - Training acc: %.4f - Training loss: %.4f - Val acc: %.4f - Val loss: %.4f - Time: %.4fs \n' % (
#     e+1,
#     run_acc/len(train_loader),
#     run_loss/len(train_loader),
#     run_acc_val/len(valid_loader),
#     run_loss_val/len(valid_loader),
#     (time.time()-timer)
#   )

#   print(print_out, end='')

#   model_name = '%s/model_ep_%d.tar' % (model_folder, e+1)
#   torch.save({
#     'epoch': e+1,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': run_loss}, 
#   model_name)

# print('training completed!')
