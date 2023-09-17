import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from REC.model.layers import LightGCNConv


class LightGCN(BaseModel):

    input_type = InputType.PAIR
    
    def __init__(self, config, data):
        super(LightGCN, self).__init__()
        self.latent_dim = config['embedding_size']  
        self.n_layers = config['n_layers']  

        self.device = config['device']
              
        self.user_num = data.user_num
        self.item_num = data.item_num
        
        self.edge_index, self.edge_weight = data.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)

        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        
       
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.store_ufeatures = None
        self.store_ifeatures = None
        self.apply(self._init_weights)
     

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
    
    
    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def computer(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings
 
    def forward(self, input):
        user, item = input
        user_all_embeddings, item_all_embeddings = self.computer()
        embed_user = user_all_embeddings[user].unsqueeze(1)
        embed_item = item_all_embeddings[item]
        score = (embed_user * embed_item).sum(-1)
        output = score.view(-1,2)
        batch_loss = -torch.mean(1e-8+torch.log(torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
      
           
    @torch.no_grad()
    def predict(self, user,features_pad):    
        embed_user = self.store_ufeatures[user]       
        scores = torch.matmul(embed_user,self.store_ifeatures.t())
        return scores

    @torch.no_grad()   
    def compute_item_all(self):
        self.store_ufeatures, self.store_ifeatures= self.computer()
        return None





