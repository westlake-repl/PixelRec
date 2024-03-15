import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

class CuratorNet(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(CuratorNet, self).__init__()
        self.embedding_size = config['embedding_size'] 
        self.hidden_size = config['hidden_size']*self.embedding_size
        self.device = config['device']
            
        self.v_feat_path = config['v_feat_path']
        v_feat = np.load(self.v_feat_path, allow_pickle=True)   
                   
        v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        v_feat[0].fill_(0)
        self.embedding = nn.Embedding.from_pretrained(v_feat, freeze=True)
        
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)

        self.feature_dim = v_feat.shape[-1]
                
        # Common section
        self.selu_common1 = nn.Linear(self.feature_dim, self.embedding_size)
        self.selu_common2 = nn.Linear(self.embedding_size, self.embedding_size)

        # Profile section
        self.maxpool = nn.AdaptiveMaxPool2d((1, self.embedding_size))
        self.avgpool = nn.AdaptiveAvgPool2d((1, self.embedding_size))
        self.selu_pu1 = nn.Linear(self.embedding_size + self.embedding_size, self.hidden_size)
        self.selu_pu2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.selu_pu3 = nn.Linear(self.hidden_size, self.embedding_size)

        # Random weight initialization
        self.reset_parameters()
     

    def reset_parameters(self):
        """Resets network weights.

        Restart network weights using a Xavier uniform distribution.
        """
        # Common section
        nn.init.xavier_uniform_(self.selu_common1.weight)
        nn.init.xavier_uniform_(self.selu_common2.weight)
        # Profile section
        nn.init.xavier_uniform_(self.selu_pu1.weight)
        nn.init.xavier_uniform_(self.selu_pu2.weight)
        nn.init.xavier_uniform_(self.selu_pu3.weight)
   
 
    def forward(self, inputs):           #inputs: user_seq, pos, neg
        profile = inputs[:, :-2]
        pi = inputs[:, -2] 
        ni = inputs[:, -1] 
        # Load embedding data
        profile = self.embedding(profile)
        pi = self.embedding(pi)
        ni = self.embedding(ni)

        # Positive item
        pi = F.selu(self.selu_common1(pi))
        pi = F.selu(self.selu_common2(pi))

        # Negative item
        ni = F.selu(self.selu_common1(ni))
        ni = F.selu(self.selu_common2(ni))

        # User profile
        profile = F.selu(self.selu_common1(profile))
        profile = F.selu(self.selu_common2(profile))
        profile = torch.cat(
            (self.maxpool(profile), self.avgpool(profile)), dim=-1
        )
        profile = F.selu(self.selu_pu1(profile))
        profile = F.selu(self.selu_pu2(profile))
        profile = F.selu(self.selu_pu3(profile))

        # x_ui > x_uj  
        profile = profile.squeeze(1)    
        x_ui = (profile*pi).sum(-1)
        x_uj = (profile*ni).sum(-1)       
        batch_loss = -torch.mean(torch.log(1e-8 + torch.sigmoid(x_ui - x_uj)))
        return batch_loss

    

    
    @torch.no_grad()
    def predict(self, user,item_feature):             
        profile = item_feature[user]              
        profile = torch.cat(
            (self.maxpool(profile), self.avgpool(profile)), dim=-1
        )
       
        profile = F.selu(self.selu_pu1(profile))
        profile = F.selu(self.selu_pu2(profile))
        profile = F.selu(self.selu_pu3(profile))  
        profile = profile.squeeze(1)
         
        score = torch.matmul(profile,item_feature.t())       
        return score

    @torch.no_grad()    # [num_item, 32]
    def compute_item_all(self):
        embed = self.embedding.weight
        embed = F.selu(self.selu_common1(embed))
        embed = F.selu(self.selu_common2(embed))
        return embed




