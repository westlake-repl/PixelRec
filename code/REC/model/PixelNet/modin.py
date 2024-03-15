import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers, SequenceAttLayer 
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.load import load_model

class MODIN(BaseModel):

    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(MODIN, self).__init__()

        # get field names and parameter value from config
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.device = config['device']    
        self.dropout_prob = config['dropout_prob']
        
        self.item_num = dataload.item_num
        self.att_list = [4 * self.embedding_size] + self.mlp_hidden_size

        self.attention = SequenceAttLayer(
            self.att_list, activation='Sigmoid', softmax_stag=False, return_seq_weight=False
        )
        
        # parameters initialization
        self.apply(self._init_weights)
        self.visual_encoder = load_model(config=config)
        # if self.pretrain_weights:
        #     self.load_weights(self.pretrain_weights)
      

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, inputs):
        #[batch,seq_len+2,3,224,224] 
        items_modal, items = inputs
        batch_size = items.shape[0]         
        item_emb = self.visual_encoder(items_modal.flatten(0,1)).view(batch_size, -1, self.embedding_size)  #[batch,seq_len+2,dim] 
        user_seq_emb = item_emb[:, :-2]
        pos_cand_embs = item_emb[:, -2] 
        neg_cand_embs = item_emb[:, -1]   

        # attention
        mask = (items[:,:-2] == 0)
        pos_user_emb = self.attention(pos_cand_embs, user_seq_emb,mask).squeeze(1)
        neg_user_emb = self.attention(neg_cand_embs, user_seq_emb,mask).squeeze(1)

        pos_score = (pos_user_emb * pos_cand_embs).sum(-1)   #[batch]
        neg_score = (neg_user_emb * neg_cand_embs).sum(-1)   #[batch]
        
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)).mean(-1) 
        return loss


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
    def compute_item(self, item):
        return self.visual_encoder(item)
