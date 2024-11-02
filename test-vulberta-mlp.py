from transformers import RobertaForSequenceClassification, RobertaConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json

import sklearn
import sklearn.metrics

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

p1, p2, p3 , n = 0, 0, 0, 0
for l in test_labels: 
  if l == 0: n += 1
  elif l == 1: p1 += 1
  elif l == 2: p2 += 1
  elif l == 3: p3 += 1

print(p1, p2, p3, n)

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

# config = RobertaConfig.from_json_file('./models/vulberta-pretrained/config.json')
# model = RobertaForSequenceClassification(config)
# checkpoint = torch.load('./models/vulberta-mlp-cwe-664/model_ep_10.tar')
# model.load_state_dict(checkpoint['model_state_dict'])

model_g = RobertaForSequenceClassification.from_pretrained('./models/vulberta-mlp-reveal')
model_664 = RobertaForSequenceClassification.from_pretrained('./models/vulberta-mlp-cwe-664/checkpoint-186350')
model_119 = RobertaForSequenceClassification.from_pretrained('./models/vulberta-mlp-cwe-119/checkpoint-143650')

model_g.to(device)
model_664.to(device)
model_119.to(device)

# vulberta_reveal = RobertaForSequenceClassification.from_pretrained('./models/vulberta-mlp-reveal')
# vulberta_cwe_664 = RobertaForSequenceClassification.from_pretrained('./models/vulberta-mlp-cwe-664/checkpoint-186350')

# class VulbertaChain(nn.Module):
  
#     def __init__(self, general_model, specific_model):
#         super(VulbertaChain, self).__init__()

#         self.general_roberta = general_model.roberta
#         self.specific_encoder = specific_model.roberta.encoder
#         self.specific_classifier = specific_model.classifier

#     def forward(self, input_ids, attention_mask=None, labels=None):
      
#         general_roberta_output = self.general_roberta(input_ids, attention_mask=attention_mask)
#         specific_encoder_output = self.specific_encoder(general_roberta_output.last_hidden_state)
#         return self.specific_classifier(specific_encoder_output.last_hidden_state)

# model = VulbertaChain(general_model=vulberta_reveal, specific_model=vulberta_cwe_664)
# model.to(device)

all_labels = []
all_probs = []
all_preds = []

model_g.eval()
model_664.eval()
model_119.eval()
with torch.no_grad():
  for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    output_g = model_g(input_ids, attention_mask=attention_mask)
    output_664 = model_664(input_ids, attention_mask=attention_mask)
    output_119 = model_119(input_ids, attention_mask=attention_mask)

    probs_g = torch.nn.functional.softmax(output_g['logits'], dim=1)
    probs_664 = torch.nn.functional.softmax(output_664['logits'], dim=1)
    probs_119 = torch.nn.functional.softmax(output_119['logits'], dim=1)

    preds_g = torch.argmax(probs_g, dim=1).tolist()
    preds_664 = torch.argmax(probs_664, dim=1).tolist()
    preds_119 = torch.argmax(probs_119, dim=1).tolist()
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
      
    all_preds += preds
    all_labels += labels.tolist()

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
