import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers, BaseFactorizationMachine
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from logging import getLogger


class FM(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(FM, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.out_size = self.mlp_hidden_size[-1] if (self.mlp_hidden_size and len(self.mlp_hidden_size)) else self.embedding_size

        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        self.item_num = dataload.item_num
        #self.user_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        
        #size_list = [self.embedding_size] + self.mlp_hidden_size + [self.embedding_size]       
        #self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    
    
    def mask_emb(self, user_seq):
        mask = user_seq != 0  
        mask = mask.float()
        
        token_seq_embedding = self.item_embedding(user_seq)  
        mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
        masked_token_seq_embedding = token_seq_embedding * mask
 
        return masked_token_seq_embedding

    
    
    def forward(self, inputs):                
        inputs = inputs[0].unsqueeze(0)
        
        inputs_embedding = self.mask_emb(inputs)
        scores = self.fm(inputs_embedding.flatten(0,1))  
        output = scores.view(-1,2) 
         
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
   

    @torch.no_grad()
    def predict(self,user_seq,item_feature):                                                
        user_embedding = self.mask_emb(user_seq)   
        user_embedding = torch.sum(user_embedding, dim=1)
        scores = torch.matmul(user_embedding,item_feature.t())
        return scores

    @torch.no_grad()    
    def compute_item_all(self):
        return self.item_embedding.weight
 
  




