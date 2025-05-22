# import pandas as pd
# import torch
# from torch import nn
# from transformers import AutoTokenizer, AutoModel
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from sklearn.metrics import (
#     f1_score, recall_score, precision_score, roc_auc_score, 
#     brier_score_loss, confusion_matrix
# )
# import json
# import argparse
# import os
# import warnings

# # Suppress huggingface_hub warning
# warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# parser = argparse.ArgumentParser(description="Evaluate HardMoEClassifier and output metrics.")
# parser.add_argument('--model-path', type=str, required=True, help='Path to the model .pth file')
# parser.add_argument('--input-dir', type=str, required=True, help='Directory containing test.csv')
# parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output.json')
# args = parser.parse_args()

# class CustomDataset(Dataset):
#     def __init__(self, df, tokenizer, text_column, label_column, max_length=512):
#         self.texts = df[text_column].tolist()
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.labels = torch.tensor(df[label_column].values, dtype=torch.long)

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].squeeze(0).long(),
#             'attention_mask': encoding['attention_mask'].squeeze(0).long(),
#             'label': self.labels[idx]
#         }

# class HardMoEClassifier(nn.Module):
#     def __init__(self, num_labels=2, dropout_prob=0.1):
#         super(HardMoEClassifier, self).__init__()
#         self.base_model = AutoModel.from_pretrained('albert-base-v2', num_labels=num_labels)
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.experts = nn.ModuleList([
#             nn.Linear(768, num_labels) for _ in range(6)
#         ])
#         self.gate = nn.Linear(768, len(self.experts))

#     def forward(self, input_ids, attention_mask):
#         outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_state = outputs.last_hidden_state
#         cls_token = hidden_state[:, 0, :]
#         cls_token = self.dropout(cls_token)
#         gate_logits = self.gate(cls_token)
#         expert_choice = torch.argmax(gate_logits, dim=1)
#         outputs = torch.zeros((input_ids.shape[0], self.experts[0].out_features)).to(input_ids.device)
#         for i, expert in enumerate(self.experts):
#             mask = expert_choice == i
#             if mask.any():
#                 expert_output = expert(cls_token[mask])
#                 outputs[mask] = expert_output
#         return outputs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
# model = HardMoEClassifier(num_labels=2).to(device)
# model.load_state_dict(torch.load(args.model_path, map_location=device))
# model.eval()

# # Debug: Print input path and verify file existence
# input_file = os.path.join(args.input_dir, "test.csv")
# print(f"Attempting to read test.csv from: {input_file}")
# if not os.path.exists(input_file):
#     print(f"Error: File {input_file} does not exist")
#     exit(1)

# test_data = pd.read_csv(input_file)
# text_column = "text"
# label_column = "label"
# test_dataset = CustomDataset(test_data, tokenizer, text_column, label_column)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# predictions = []
# probabilities = []
# true_labels = []

# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
#         outputs = model(input_ids, attention_mask)
#         probs = torch.softmax(outputs, dim=1)
#         predicted_labels = outputs.argmax(dim=1).cpu().numpy()
#         predictions.extend(predicted_labels)
#         probabilities.extend(probs.cpu().numpy())
#         true_labels.extend(labels.cpu().numpy())

# predictions = np.array(predictions)
# true_labels = np.array(true_labels)
# probabilities = np.array(probabilities)
# num_classes = 2

# val_true_one_hot = np.zeros((true_labels.size, num_classes))
# val_true_one_hot[np.arange(true_labels.size), true_labels] = 1
# roc_auc_macro = roc_auc_score(val_true_one_hot, probabilities, multi_class='ovr', average='macro')

# brier = np.mean([
#     brier_score_loss((true_labels == i).astype(int), probabilities[:, i])
#     for i in range(num_classes)
# ])
# brier_complement = 1 - brier

# correct = (predictions == true_labels).astype(int)
# unanswered = np.zeros_like(correct)
# c_at_1 = (sum(correct) + sum(unanswered) * sum(correct) / len(correct)) / len(correct)

# f1 = f1_score(true_labels, predictions, average='macro')
# precision = precision_score(true_labels, predictions, average='macro')
# recall = recall_score(true_labels, predictions, average='macro')
# f05u = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall + 1e-10)
# confusion = confusion_matrix(true_labels, predictions).tolist()
# mean_metric = np.mean([roc_auc_macro, brier_complement, c_at_1, f1, f05u])

# json_data = {
#     "roc-auc": round(roc_auc_macro, 4),
#     "brier": round(brier_complement, 4),
#     "c@1": round(c_at_1, 4),
#     "f1": round(f1, 4),
#     "f05u": round(f05u, 4),
#     "mean": round(mean_metric, 4),
#     "confusion": confusion
# }

# os.makedirs(args.output_dir, exist_ok=True)
# with open(os.path.join(args.output_dir, "output.json"), "w") as f:
#     json.dump(json_data, f, indent=4)

# print("Test Metrics JSON:", json_data)



import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score, 
    brier_score_loss, confusion_matrix
)
import json
import argparse
import os
import warnings

# Suppress huggingface_hub warning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

parser = argparse.ArgumentParser(description="Evaluate HardMoEClassifier and output metrics.")
parser.add_argument('--model-path', type=str, required=True, help='Path to the model .pth file')
parser.add_argument('--input-file', type=str, required=True, help='Path to the test CSV file')
parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output.json')
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, text_column, max_length=512):
        self.texts = df[text_column].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0).long(),
            'attention_mask': encoding['attention_mask'].squeeze(0).long(),
        }

class HardMoEClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_prob=0.1):
        super(HardMoEClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained('albert-base-v2', num_labels=num_labels)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.experts = nn.ModuleList([
            nn.Linear(768, num_labels) for _ in range(6)
        ])
        self.gate = nn.Linear(768, len(self.experts))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        cls_token = hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        gate_logits = self.gate(cls_token)
        expert_choice = torch.argmax(gate_logits, dim=1)
        outputs = torch.zeros((input_ids.shape[0], self.experts[0].out_features)).to(input_ids.device)
        for i, expert in enumerate(self.experts):
            mask = expert_choice == i
            if mask.any():
                expert_output = expert(cls_token[mask])
                outputs[mask] = expert_output
        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
model = HardMoEClassifier(num_labels=2).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Debug: Print input file path and verify file existence
input_file = args.input_file
print(f"Attempting to read test jsonl from: {input_file}")
if not os.path.exists(input_file):
    print(f"Error: File {input_file} does not exist")
    exit(1)

test_data = df = pd.read_json(input_file, lines=True)
text_column = "text"
test_dataset = CustomDataset(test_data, tokenizer, text_column)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

predictions = []
probabilities = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        predicted_labels = outputs.argmax(dim=1).cpu().numpy()
        predictions.extend(predicted_labels)
        probabilities.extend(probs.cpu().numpy())

predictions = np.array(predictions)
true_labels = np.array(true_labels)
probabilities = np.array(probabilities)
num_classes = 2

val_true_one_hot = np.zeros((true_labels.size, num_classes))
val_true_one_hot[np.arange(true_labels.size), true_labels] = 1
roc_auc_macro = roc_auc_score(val_true_one_hot, probabilities, multi_class='ovr', average='macro')

brier = np.mean([
    brier_score_loss((true_labels == i).astype(int), probabilities[:, i])
    for i in range(num_classes)
])
brier_complement = 1 - brier

correct = (predictions == true_labels).astype(int)
unanswered = np.zeros_like(correct)
c_at_1 = (sum(correct) + sum(unanswered) * sum(correct) / len(correct)) / len(correct)

f1 = f1_score(true_labels, predictions, average='macro')
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f05u = (1 + 0.5 ** 2) * (precision * recall) / ((0.5 ** 2 * precision) + recall + 1e-10)
confusion = confusion_matrix(true_labels, predictions).tolist()
mean_metric = np.mean([roc_auc_macro, brier_complement, c_at_1, f1, f05u])

json_data = {
    "roc-auc": round(roc_auc_macro, 4),
    "brier": round(brier_complement, 4),
    "c@1": round(c_at_1, 4),
    "f1": round(f1, 4),
    "f05u": round(f05u, 4),
    "mean": round(mean_metric, 4),
    "confusion": confusion
}

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "output.json"), "w") as f:
    json.dump(json_data, f, indent=4)

print("Test Metrics JSON:", json_data)