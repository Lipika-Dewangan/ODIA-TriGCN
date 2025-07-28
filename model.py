import torch
import torch.nn as nn
from layers import GCNLayer, BiaffineAttention

class OdiaTriGCN(nn.Module):
    def __init__(self, config, bert_model):
        super(OdiaTriGCN, self).__init__()
        self.bert = bert_model
        self.gcn_dis = GCNLayer(config['hidden_dim'], config['hidden_dim'])
        self.gcn_syn = GCNLayer(config['hidden_dim'], config['hidden_dim'])
        self.biaffine = BiaffineAttention(config['hidden_dim'])
        self.mlp = nn.Sequential(
            nn.Linear(config['hidden_dim'] * 3, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        self.classifier = nn.Linear(config['hidden_dim'] * 2, config['num_classes'])

    def forward(self, input_ids, attention_mask, adj_dis, adj_syn):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h_dis = self.gcn_dis(bert_out, adj_dis)
        h_syn = self.gcn_syn(bert_out, adj_syn)
        h_bfa = self.biaffine(bert_out)
        h_cat = torch.cat([h_dis, h_syn, h_bfa], dim=-1)
        h_out = self.mlp(h_cat)
        return h_out
