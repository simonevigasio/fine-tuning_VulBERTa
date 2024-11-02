import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaModel

import sklearn

import numpy
import pandas

import json

import time

import os

vocab_size = 50000
embed_size = vocab_size + 2
embed_dim = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_valid_enc = open('./datasets/diversevul-cwe-119/encodings/valid.json', 'r')
file_train_enc = open('./datasets/diversevul-cwe-119/encodings/train.json', 'r')

valid_enc = json.load(file_valid_enc)
train_enc = json.load(file_train_enc)

file_valid_data = open('./datasets/diversevul-cwe-119/data/valid.json', 'r')
file_train_data = open('./datasets/diversevul-cwe-119/data/train.json', 'r')

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
      if 'CWE-664' in data['pillar']:
        label = 2
        if 'cwe_119' in data:
          label = 3
      
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

valid_loader = DataLoader(valid, batch_size=128, shuffle=True)
train_loader = DataLoader(train, batch_size=128, shuffle=True)

class vulberta_cnn(nn.Module):
  
  def __init__(self, embed_size, embed_dim):
    super(vulberta_cnn, self).__init__()

    pretrained_weights = RobertaModel.from_pretrained('./models/vulberta-pretrained').embeddings.word_embeddings.weight
    self.embed = nn.Embedding.from_pretrained(pretrained_weights, freeze=True, padding_idx=1)

    self.conv_1 = nn.Conv1d(in_channels=embed_dim, out_channels=200, kernel_size=3)
    self.conv_2 = nn.Conv1d(in_channels=embed_dim, out_channels=200, kernel_size=4)
    self.conv_3 = nn.Conv1d(in_channels=embed_dim, out_channels=200, kernel_size=5)

    self.dropout = nn.Dropout(0.5)

    self.fully_connected_1 = nn.Linear(600, 256)
    self.fully_connected_2 = nn.Linear(256, 128)
    self.fully_connected_3 = nn.Linear(128, 4)

  def forward(self, x):
    x = self.embed(x)
    x = x.permute(0, 2, 1)

    x1 = F.relu(self.conv_1(x))
    x2 = F.relu(self.conv_2(x))
    x3 = F.relu(self.conv_3(x))

    x1 = F.max_pool1d(x1, x1.shape[2])
    x2 = F.max_pool1d(x2, x2.shape[2])
    x3 = F.max_pool1d(x3, x3.shape[2])

    x = torch.cat([x1 ,x2, x3], dim=1)

    x = x.flatten(1)

    x = self.dropout(x)

    x = F.relu(self.fully_connected_1(x))
    x = F.relu(self.fully_connected_2(x))
    x = self.fully_connected_3(x)

    return (x)
  
model = vulberta_cnn(embed_size, embed_dim)
model.embed.weight.data[1] = torch.zeros(embed_dim)
model.to(device)

print(model)
print('Num of trainable param', sum(p.numel() for p in model.parameters() if p.requires_grad))

class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=numpy.array([0, 1, 2, 3]), y=train_labels)
class_weights = torch.FloatTensor([class_weights[0], class_weights[1], class_weights[2], class_weights[3]])
criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = criterion.to(device)

optimizer = Adam(model.parameters(), lr=0.0005)

model_folder = './models/vulberta-cnn-cwe-664-119'
if not os.path.exists(model_folder):
  os.makedirs(model_folder)

def get_class(x):
  return(x.index(max(x)))

print('training started...')

epoch = 20

for e in range(epoch):
  timer = time.time()

  run_acc, run_loss = 0, 0
  model.train()

  for batch in train_loader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()
    output = model(input_ids)

    loss = criterion(output, labels)
    loss.backward()

    optimizer.step()

    probs = pandas.Series(output.tolist())
    preds = probs.apply(get_class)
    preds.reset_index(drop=True, inplace=True)
    value_counts = pandas.value_counts(preds == labels.tolist())
    try:
      acc = value_counts[1]/len(labels.tolist()) 
    except:
      if value_counts.index[0]==False:
        acc = 0
      else:
        acc = 1

    run_acc += acc
    run_loss += loss.item()

  with torch.no_grad():
    run_acc_val, run_loss_val = 0, 0
    model.eval()

    for batch in valid_loader:
      input_ids_val = batch['input_ids'].to(device)
      labels_val = batch['labels'].to(device)

      output_val = model(input_ids_val)

      loss_val = criterion(output_val, labels_val)

      probs_val = pandas.Series(output_val.tolist())
      preds_val = probs_val.apply(get_class)
      preds_val.reset_index(drop=True, inplace=True)
      value_counts_val = pandas.value_counts(preds_val == labels_val.tolist())
      try:
        acc_val = value_counts_val[1]/len(labels_val.tolist()) 
      except: 
        if value_counts_val.index[0]==False:
          acc_val = 0
        else: 
          acc_val = 1

      run_acc_val += acc_val
      run_loss_val += loss_val.item()

  print_out = 'Epoch %d - Training acc: %.4f - Training loss: %.4f - Val acc: %.4f - Val loss: %.4f - Time: %.4fs \n' % (
    e+1,
    run_acc/len(train_loader),
    run_loss/len(train_loader),
    run_acc_val/len(valid_loader),
    run_loss_val/len(valid_loader),
    (time.time()-timer)
  )

  print(print_out, end='')

  model_name = '%s/model_ep_%d.tar' % (model_folder, e+1)
  torch.save({
    'epoch': e+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': run_loss}, 
  model_name)

print('training completed!')
