from transformers import RobertaModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sklearn
import sklearn.metrics

import pandas as pd
import numpy as np

import json

vocab_size = 50000
embed_size = vocab_size + 2
embed_dim = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

file_test_enc = open('./datasets/diversevul-2.0/enc/test.json', 'r')
test_enc = json.load(file_test_enc)

file_test_data = open('./datasets/diversevul-2.0/data/test.json', 'r')
test_data = json.load(file_test_data)

assert len(test_data) == len(test_enc['input_ids']) == len(test_enc['attention_mask'])

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

test_labels = process_labels(test_data)

p, n = 0, 0
for l in test_labels: 
  if l == 0: n += 1
  else: p += 1

print('perc:', p/(n+p))

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
  
test = VulnerabilityDetectiondDataset(encodings=test_enc, labels=test_labels)
test_loader = DataLoader(test, batch_size=1)

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
    self.fully_connected_3 = nn.Linear(128, 2)

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
  
model_g = vulberta_cnn(embed_size, embed_dim)
model_664 = vulberta_cnn(embed_size, embed_dim)
model_119 = vulberta_cnn(embed_size, embed_dim)

checkpoint_g = torch.load('./models/vulberta-cnn-generic/model_ep_20.tar')
checkpoint_664 = torch.load('./models/vulberta-cnn-cwe-664/model_ep_20.tar')
checkpoint_119 = torch.load('./models/vulberta-cnn-cwe-119/model_ep_20.tar')

model_g.load_state_dict(checkpoint_g['model_state_dict'])
model_664.load_state_dict(checkpoint_664['model_state_dict'])
model_119.load_state_dict(checkpoint_119['model_state_dict'])

model_g.to(device)
model_664.to(device)
model_119.to(device)

# class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=test_labels)
# class_weights = torch.FloatTensor([class_weights[0], class_weights[1]])
# criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = criterion.to(device)

all_labels = []
all_probs = []
all_preds = []

model_g.eval()
model_664.eval()
model_119.eval()
with torch.no_grad():
    run_loss = 0

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        output_g = model_g(input_ids).squeeze(1)
        output_664 = model_664(input_ids).squeeze(1)
        output_119 = model_119(input_ids).squeeze(1)

        # loss = criterion(output, labels)
        # run_loss += loss.item()

        preds_g = torch.argmax(output_g, dim=1)
        preds_664 = torch.argmax(output_664, dim=1)
        preds_119 = torch.argmax(output_119, dim=1)

        preds_g = preds_g.tolist()
        preds_664 = preds_664.tolist()
        preds_119 = preds_119.tolist()
        assert len(preds_g) == len(preds_664) == len(preds_119)

        preds = []
        for i in range(len(preds_g)):
          pred = 0
          if preds_g[i] == 1: 
            pred = 1
          if preds_664[i] == 1:
            pred = 2
          if preds_119[i] == 1:
            pred = 3
          preds.append(pred)

        labels = labels.tolist()
        assert len(preds) == len(labels)

        all_preds += preds
        all_labels += labels

confusion = sklearn.metrics.confusion_matrix(y_true=all_labels, y_pred=all_preds)
print(confusion)

# tn, fp, fn, tp = confusion.ravel()
tn = confusion[0][0]
fp = confusion[0][1] + confusion[0][2] + confusion[0][3] + confusion[1][2] + confusion[1][3] + confusion[2][3]
fn = confusion[1][0] + confusion[2][0] + confusion[3][0]
tp = confusion[1][1] + confusion[2][2] + confusion[3][3] + confusion[2][1] + confusion[3][1] + confusion[3][2]

print('tn', tn)
print('fp', fp)
print('fn', fn)
print('tp', tp)

# print('Accuracy', str(sklearn.metrics.accuracy_score(y_true=all_labels, y_pred=all_preds)))
# print('Precision', str(sklearn.metrics.precision_score(y_true=all_labels, y_pred=all_preds)))
# print('Recall', str(sklearn.metrics.recall_score(y_true=all_labels, y_pred=all_preds)))
# print('F-measure', str(sklearn.metrics.f1_score(y_true=all_labels, y_pred=all_preds)))
