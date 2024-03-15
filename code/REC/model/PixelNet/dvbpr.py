import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F



class DVBPR(BaseModel):
    input_type = InputType.PAIR
    def __init__(self, config, dataload):
        super(DVBPR, self).__init__()
     
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size'] // 2
    
        self.device = config['device']
              
        self.user_num = dataload.user_num
        self.item_num = dataload.item_num
        # CNN for learned image features
        
        self.visual_encoder = CNNF(hidden_dim=self.embedding_size, dropout=self.dropout_prob)  # CNN-F is a smaller CNN
       
        # Visual latent preference (theta)
        self.theta_users = nn.Embedding(self.user_num, self.embedding_size)

        # Latent factors (gamma)
        self.gamma_users = nn.Embedding(self.user_num, self.embedding_size)
        self.gamma_items = nn.Embedding(self.item_num, self.embedding_size)
        
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        # Random weight initialization
        self.reset_parameters()

    
    
    def reset_parameters(self):
        """ Restart network weights using a Xavier uniform distribution. """
        if isinstance(self.visual_encoder, CNNF):
            self.visual_encoder.reset_parameters()
        nn.init.uniform_(self.theta_users.weight)  # Visual factors (theta)
        nn.init.uniform_(self.gamma_users.weight)  # Visual factors (theta)
        nn.init.uniform_(self.gamma_items.weight)  # Visual factors (theta)
    
    
    def forward(self, inputs):
        user, item_id, item_modal = inputs
        embed_id_user = self.gamma_users(user).unsqueeze(1)   
        embed_id_item = self.gamma_items(item_id)  

        embed_modal_user = self.theta_users(user).unsqueeze(1)
        embed_modal_item = self.visual_encoder(item_modal).view(user.shape[0], -1, self.embedding_size)   #[5,2,32]

        score = (embed_id_user * embed_id_item).sum(-1) + \
            (embed_modal_user * embed_modal_item).sum(-1) 
                    
        output = score.view(-1,2)    
        batch_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss



    @torch.no_grad()
    def predict(self, user,item_feature): 
        embed_id_user = self.gamma_users(user)  
        embed_id_item = self.gamma_items.weight   

        embed_modal_user = self.theta_users(user)
        
        score = torch.matmul(embed_id_user,embed_id_item.t()) + \
        torch.matmul(embed_modal_user,item_feature.t()) 
        return score

    @torch.no_grad()
    def compute_item(self, item):
        return self.visual_encoder(item)












class CNNF(nn.Module):
    """CNN-F network"""
    def __init__(self, hidden_dim=2048, fc_dim=512, weights=None, dropout=0.5):
        super(CNNF, self).__init__()
        self.hidden_dim = hidden_dim

        if weights is None:
            weights = {
                # conv layers: ((c_in, c_out, stride (square)), custom stride)
                'cnn': [([3, 64, 11], [1, 4]),
                        ([64, 256, 5], None),
                        ([256, 256, 3], None),
                        ([256, 256, 3], None),
                        ([256, 256, 3], None)],
                    
                # fc layers: n_in, n_out
                'fc': [[256*22*2, fc_dim],  # original: 256*7*7 -> 4096
                    [fc_dim, fc_dim],
                    [fc_dim, self.hidden_dim]]
            }

        self.convs = nn.ModuleList([nn.Conv2d(*params, padding_mode='replicate', stride=stride if stride else 1)
                                    for params, stride in weights['cnn']])
        
        self.fcs = nn.ModuleList([nn.Linear(*params) for params in weights['fc']])
        self.maxpool2d = nn.MaxPool2d(2)
        self.maxpool_idxs = [True, True, False, False, True]  # CNN layers to maxpool
        self.dropout = nn.Dropout(p=dropout)        
        self.layer_params = weights

    def forward(self, x):
        x = torch.reshape(x, shape=[-1, 3, 224, 224])

        # convolutional layers
        for cnn_layer, apply_maxpool in zip(self.convs, self.maxpool_idxs):
            x = F.relu(cnn_layer(x))
            # notable difference: original TF implementation has "SAME" padding
            x = self.maxpool2d(x) if apply_maxpool else x

        # fully connected layers
        x = torch.reshape(x, shape=[-1, self.layer_params['fc'][0][0]])
        for fc_layer in self.fcs:
            x = F.relu(fc_layer(x))
            x = self.dropout(x)

        return x

    def reset_parameters(self):
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)