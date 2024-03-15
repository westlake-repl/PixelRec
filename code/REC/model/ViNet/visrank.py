import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np


class VISRANK(nn.Module):
    input_type = InputType.PAIR
    def __init__(self, config, dataload):
        super(VISRANK, self).__init__()
              
        
        self.method = config['method']
        if self.method == 'average_top_k':
            self.k = config['top_num']
        elif self.method == 'maximum':
            self.k = 1
        else:
            self.k = None
        v_feat_path = config['v_feat_path']
        self.device = config['device']
        v_feat = np.load(v_feat_path, allow_pickle=True)   
                   
        self.v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        
        self.module = None
        self.placeholder = nn.Parameter(torch.zeros(0, requires_grad=True))
     


    def forward(self, inputs):
        pass
    
   
    @torch.no_grad()      #set batch=1 
    def predict(self, user,item_feature):

        user = user[-50:]  # due to the limited of GPU memory RN50:-50  resnet50:-30

    
        seq_feat = self.v_feat[user]
        possible_items = torch.cosine_similarity(seq_feat.unsqueeze(1),self.v_feat.unsqueeze(0),dim=-1)
        
        seq_len = len(user)
        if self.method == 'average_top_k':
            k = min(self.k, seq_len)
        elif self.method == 'maximum':
            k = 1
        else:
            k = seq_len
        values, _ = torch.topk(possible_items, k = k, dim=0)
        scores = values.mean(0)
        scores[0] = -np.inf       
        return scores

    @torch.no_grad()    
    def compute_item_all(self):
        return None




