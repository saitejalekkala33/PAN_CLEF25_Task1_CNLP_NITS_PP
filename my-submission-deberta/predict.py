import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

parser = argparse.ArgumentParser(description="Evaluate DeBERTa-V3_Large HardMoEClassifier and output metrics.")
parser.add_argument('--model-path', type=str, required=True, help='Path to the model .pth file')
parser.add_argument('--input-file', type=str, required=True, help='Path to the test CSV file')
parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output.json')
args = parser.parse_args()

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, text_column, max_length=512):
        self.ids = df["id"].tolist()
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
            'id': self.ids[idx],
            'input_ids': encoding['input_ids'].squeeze(0).long(),
            'attention_mask': encoding['attention_mask'].squeeze(0).long(),
        }

class HardMoEClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_prob=0.1):
        super(HardMoEClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained('microsoft/deberta-v3-large')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.experts = nn.ModuleList([
            nn.Linear(1024, num_labels) for _ in range(6)
        ])
        self.gate = nn.Linear(1024, len(self.experts))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        cls_token = hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        gate_logits = self.gate(cls_token)
        expert_choice = torch.argmax(gate_logits, dim=1)
        output = torch.zeros((input_ids.shape[0], self.experts[0].out_features)).to(input_ids.device)
        for i, expert in enumerate(self.experts):
            mask = expert_choice == i
            if mask.any():
                expert_output = expert(cls_token[mask])
                output[mask] = expert_output
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
model = HardMoEClassifier(num_labels=2).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

input_file = args.input_file
print(f"Attempting to read test jsonl from: {input_file}")
if not os.path.exists(input_file):
    print(f"Error: File {input_file} does not exist")
    exit(1)

test_data = df = pd.read_json(input_file, lines=True)
test_dataset = CustomDataset(test_data, tokenizer, "text")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "output.jsonl"), "w") as f:
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['id']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            predicted_labels = outputs.argmax(dim=1).cpu().numpy()
            for id, score in zip(ids, predicted_labels):
                json.dump({'id': id, 'label': float(score)}, f)
                f.write('\n')