import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel

class DSSM(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(DSSM, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.out_size = self.mlp_hidden_size[-1] if len(self.mlp_hidden_size) else self.embedding_size

        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        self.item_num = dataload.item_num
        #self.user_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.user_embedding = self.item_embedding
        #size_list = [self.embedding_size] + self.mlp_hidden_size + [self.embedding_size]
        size_list = self.mlp_hidden_size       
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        
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

    
    
    def avg_emb(self, user_seq):
        mask = user_seq != 0 
        mask = mask.float()
        value_cnt = torch.sum(mask, dim=1, keepdim=True)  
        token_seq_embedding = self.user_embedding(user_seq)  
        mask = mask.unsqueeze(2).expand_as(token_seq_embedding)
        masked_token_seq_embedding = token_seq_embedding * mask.float()
        result = torch.sum(masked_token_seq_embedding, dim=-2)  
        user_embedding = torch.div(result, value_cnt + 1e-8)
        return user_embedding

    
    
    def forward(self, inputs):         
        inputs = inputs[0].unsqueeze(0)
        user_seq = inputs[:, :-2]      
        target_item = inputs[:, -2:]  
        user_embedding = self.avg_emb(user_seq)
        user_embedding = self.mlp_layers(user_embedding).unsqueeze(1)   
        item_embedding = self.item_embedding(target_item)        
        score = (user_embedding * item_embedding).sum(-1)    
        output = score.view(-1,2)         
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
   

    #如果concate的话，就是和顺序有关的，如果直接取mean或者add，就和顺序无关
    @torch.no_grad()
    def predict(self,user_seq,item_feature):                                                
        user_embedding = self.avg_emb(user_seq)   
        user_embedding = self.mlp_layers(user_embedding)
        scores = torch.matmul(user_embedding,item_feature.t())
        return scores

    @torch.no_grad()  
    def compute_item_all(self):
        return self.item_embedding.weight
 
        #return torch.arange(0,self.n_items).to(self.device)




