import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from torch import nn

class HardMoEClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_prob=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained('microsoft/deberta-v3-large')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.experts = nn.ModuleList([nn.Linear(1024, num_labels) for _ in range(6)])
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

AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = hf_hub_download(
    repo_id='saiteja33/PAN25_1_DeBERTaV3Large_HardMoE',
    filename='DeBERTav3L_HardMoE_Task1.pth',
    repo_type='model'
)

model = HardMoEClassifier(num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=device))
torch.save(model.state_dict(), '/app/DeBERTav3L_HardMoE_Task1.pth')
