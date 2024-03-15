import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np
from torch.nn.init import xavier_normal_, constant_

class VBPR(BaseModel):
    input_type = InputType.PAIR
    def __init__(self, config, dataload):
        super(VBPR, self).__init__()
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size'] // 2
    
        self.device = config['device']
              
        self.user_num = dataload.user_num
        self.item_num = dataload.item_num
        
        self.v_feat_path = config['v_feat_path']
        v_feat = np.load(self.v_feat_path, allow_pickle=True)   
                   
        self.v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)

        self.feature_dim = self.v_feat.shape[-1]
                
        # define layers and loss
        self.feature_projection = nn.Linear(self.feature_dim, self.embedding_size, bias=False)
        self.bias_projection = nn.Linear(self.feature_dim, 1, bias=False) 
        self.user_id_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.item_num, self.embedding_size)
        
        self.user_modal_embedding = nn.Embedding(self.user_num, self.embedding_size)

        #self.visual_bias = nn.Parameter(torch.tensor(0.0))
        #self.user_bias = nn.Parameter(torch.tensor(0.0))
        #self.item_bias = nn.Parameter(torch.tensor(0.0))
        #self.global_bias = nn.Parameter(torch.tensor(0.0))
        #self.loss = BPRLoss()
        # parameters initialization
        self.apply(self._init_weights)
     

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
   
 
    def forward(self, inputs):
        user, item = inputs
        embed_id_user = self.user_id_embedding(user).unsqueeze(1)   
        embed_id_item = self.item_id_embedding(item)   

        embed_modal_user = self.user_modal_embedding(user).unsqueeze(1)
        embed_modal_item = self.feature_projection(self.v_feat[item])  


        score = (embed_id_user * embed_id_item).sum(-1) + (embed_modal_user * embed_modal_item).sum(-1)  \
            + self.bias_projection(self.v_feat[item]).squeeze(-1)
            #self.global_bias + self.user_bias + self.item_bias 
        
        output = score.view(-1,2)    
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss

    

    
    @torch.no_grad()
    def predict(self, user,item_feature):
        embed_id_user = self.user_id_embedding(user) 
        embed_id_item = self.item_id_embedding.weight 

        embed_modal_user = self.user_modal_embedding(user)
       
     
        score = torch.matmul(embed_id_user,embed_id_item.t()) + \
        torch.matmul(embed_modal_user,item_feature.t()) + \
            self.total_visual_bias

        
        return score

    @torch.no_grad()    
    def compute_item_all(self):
        embed = self.feature_projection(self.v_feat)
        self.total_visual_bias = self.bias_projection(self.v_feat).squeeze(-1)
        return embed




