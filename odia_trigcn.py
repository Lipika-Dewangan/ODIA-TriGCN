# Odia-TriGCN: Aspect Sentiment Triplet Extraction for Low-Resource Odia


## main.py


import torch
from model.trigcn import OdiaTriGCN
from utils.evaluation import evaluate_model
from utils.grid_decoder import decode_triplets
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

# Load config
def load_config():
    import yaml
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    config = load_config()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = OdiaTriGCN(config).to("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy dataloader for structure (replace with real data loader)
    dummy_input = [tokenizer("ଏହି ସାଡ଼ୀ ର ରଙ୍ଗ ବହୁତ ଆକର୍ଷଣୀୟ ଥିଲା", return_tensors="pt")]
    model.eval()
    with torch.no_grad():
        for item in dummy_input:
            output = model(**item)
            triplets = decode_triplets(output)
            print("Predicted Triplets:", triplets)


## model/trigcn.py


import torch
import torch.nn as nn
from transformers import AutoModel
from model.gcn_layers import SyntacticGCN, BiaffineGCN, TreeGCN

class OdiaTriGCN(nn.Module):
    def __init__(self, config):
        super(OdiaTriGCN, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-multilingual-cased")
        hidden_size = config['model']['hidden_dim']

        self.biaffine = BiaffineGCN(hidden_size)
        self.syntax = SyntacticGCN(hidden_size)
        self.tree = TreeGCN(hidden_size)

        self.grid_classifier = nn.Linear(hidden_size, 3)  # aspect-opinion grid tagging
        self.dropout = nn.Dropout(config['model']['dropout'])

    def forward(self, input_ids, attention_mask, **kwargs):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]  # [B, T, H]

        biaffine_out = self.biaffine(bert_output)
        syntax_out = self.syntax(bert_output)
        tree_out = self.tree(bert_output)

        fused = (biaffine_out + syntax_out + tree_out) / 3
        fused = self.dropout(fused)
        logits = self.grid_classifier(fused)

        return logits



