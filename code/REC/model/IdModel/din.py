import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers, SequenceAttLayer 
from REC.utils import InputType
from REC.model.basemodel import BaseModel

class DIN(BaseModel):

    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(DIN, self).__init__()

        # get field names and parameter value from config
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = config['device']    
        self.dropout_prob = config['dropout_prob']
        
        self.item_num = dataload.item_num
        
        self.dnn_list = [3 * self.embedding_size] + self.mlp_hidden_size
        self.att_list = [4 * self.embedding_size] + self.mlp_hidden_size
       
        self.attention = SequenceAttLayer(
            self.att_list, activation='Sigmoid', softmax_stag=False, return_seq_weight=False
        )
        
        #self.dnn_mlp_layers = MLPLayers(self.dnn_list, activation='Dice', dropout=self.dropout_prob, bn=True)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        #self.dnn_predict_layers = nn.Linear(self.mlp_hidden_size[-1], 1)
        #self.criterion = nn.BCEWithLogitsLoss()
        # parameters initialization
        self.apply(self._init_weights)
      

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    
    
    def get_scores(self, cand_embs, user_seq_emb, mask):
        user_emb = self.attention(cand_embs, user_seq_emb,mask).squeeze(1)
        # din_in = torch.cat([user_emb, cand_embs, user_emb * cand_embs], dim=-1)
        # din_out = self.dnn_mlp_layers(din_in)
        # scores = self.dnn_predict_layers(din_out).squeeze(1)
        scores = (user_emb*cand_embs).sum(-1)
        return scores
    
    
    def forward(self, items):
        #[batch,seq_len+2]          
        item_emb = self.item_embedding(items)  #[batch,seq_len+2,dim] 
        user_seq_emb = item_emb[:, :-2]
        pos_cand_embs = item_emb[:, -2] 
        neg_cand_embs = item_emb[:, -1]   

        # attention
        mask = (items[:,:-2] == 0)
        # pos_user_emb = self.attention(pos_cand_embs, user_seq_emb,mask).squeeze(1)
        # neg_user_emb = self.attention(neg_cand_embs, user_seq_emb,mask).squeeze(1)

        # pos_score = (pos_user_emb * pos_cand_embs).sum(-1)   #[batch]
        # neg_score = (neg_user_emb * neg_cand_embs).sum(-1)   #[batch]
        pos_score = self.get_scores(pos_cand_embs, user_seq_emb, mask)
        neg_score = self.get_scores(neg_cand_embs, user_seq_emb, mask)
        
        # pos_labels, neg_labels = torch.ones(pos_score.shape).to(self.device), torch.zeros(neg_score.shape).to(self.device)
        
        # loss_1 = self.criterion(pos_score, pos_labels) 
        # loss_2 = self.criterion(neg_score, neg_labels)
        # loss = loss_1 + loss_2 
        MBAloss = 0.01 * torch.norm(item_emb, 2) / item_emb.shape[0]
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)).mean(-1) 
        return loss + MBAloss




    @torch.no_grad()
    def predict(self, item_seq, item_feature):
             
        #[batch,item_num, seq_len+1]
        batch_size = item_seq.shape[0]
        item_seq = item_seq.flatten(0,1)   #[batch*item_num, seq_len+1]       
        item_emb = item_feature[item_seq]  #[batch*item_num, seq_len+1,dim]
        user_seq_emb = item_emb[:, :-1]
        cand_emb = item_emb[:,-1] 
        

        # attention
        mask = (item_seq[:,:-1] == 0)
        user_emb = self.attention(cand_emb, user_seq_emb, mask).squeeze(1) #[batch*item_num,dim]
      
        user_emb = user_emb.view(batch_size, self.item_num, self.embedding_size) #[batch,item_num,dim]
        scores = (user_emb*item_feature).sum(-1)  # [B n_items]
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight
