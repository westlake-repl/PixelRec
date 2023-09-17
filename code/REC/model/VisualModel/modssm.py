import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.load import load_model

class MODSSM(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(MODSSM, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']

        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        self.item_num = dataload.item_num
        #self.user_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.visual_encoder = load_model(config=config)
            
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)




    
    
    def avg_emb(self, mask, token_seq_embedding):
        mask = mask.float()
        value_cnt = torch.sum(mask, dim=1, keepdim=True) 
        mask = mask.unsqueeze(2).expand_as(token_seq_embedding)
        masked_token_seq_embedding = token_seq_embedding * mask.float()
        result = torch.sum(masked_token_seq_embedding, dim=-2)  
        user_embedding = torch.div(result, value_cnt + 1e-8)
        return user_embedding

    
    
    def forward(self, inputs):  
        items_index,all_item_modal = inputs
        mask = items_index[:, :-2] != 0    
        all_item_embs = self.visual_encoder(all_item_modal)
        input_item_embs = all_item_embs[items_index, :] 

        user_embedding = input_item_embs[:, :-2, :]     
        item_embedding = input_item_embs[:, -2:,:]  
        user_embedding = self.avg_emb(mask, user_embedding).unsqueeze(1)
        score = (user_embedding * item_embedding).sum(-1)    
        output = score.view(-1,2)         
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
   

    @torch.no_grad()
    def predict(self,user_seq,item_feature):
        mask = user_seq != 0
        input_embs = item_feature[user_seq]                                                
        user_embedding = self.avg_emb(mask,input_embs)   
        scores = torch.matmul(user_embedding,item_feature.t())
        return scores
    
    
    @torch.no_grad()   
    def compute_item(self,item):
        return self.visual_encoder(item)
 





