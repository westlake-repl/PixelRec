import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from REC.utils import InputType
from REC.model.basemodel import BaseModel



class GRU4Rec(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, data):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size'] * config['embedding_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']


        self.user_num = data.user_num
        self.item_num = data.item_num
        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)


        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, inputs):
        items, masked_index = inputs
        
        item_emb = self.item_embedding(items) #[batch, 2, max_seq_len+1, dim]
        pos_items_embs = item_emb[:, 0, :]  #[batch, max_seq_len+1, dim]
        neg_items_embs = item_emb[:, 1, :]   #[batch, max_seq_len+1, dim]

        input_emb = pos_items_embs[:, :-1, :]       #[batch, max_seq_len, dim]
        target_pos_embs = pos_items_embs[:, 1:, :]  #[batch, max_seq_len, dim]
        target_neg_embs = neg_items_embs[:, 1:, :] #[batch, max_seq_len, dim]

        input_emb_dropout = self.emb_dropout(input_emb)
        gru_output, _ = self.gru_layers(input_emb_dropout)
        gru_output = self.dense(gru_output)

        pos_score = (gru_output * target_pos_embs).sum(-1)   #[batch, max_seq_len-1]
        neg_score = (gru_output * target_neg_embs).sum(-1)   #[batch, max_seq_len-1]
        
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)*masked_index).sum(-1) 
        return loss.mean(-1)


    @torch.no_grad()
    def predict(self, item_seq, item_feature):
           
        item_emb = item_feature[item_seq]

        item_seq_emb_dropout = self.emb_dropout(item_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)       
        hidden = gru_output[:, -1]                     
        scores = torch.matmul(hidden, item_feature.t())  # [B n_items]
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight



