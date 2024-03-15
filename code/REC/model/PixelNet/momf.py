import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from REC.model.layers import MLPLayers
from REC.model.load import load_model
from REC.utils import InputType
from REC.model.basemodel import BaseModel

class MOMF(BaseModel):

    input_type = InputType.PAIR

    def __init__(self, config, data):
        super(MOMF, self).__init__()

        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.out_size = self.mlp_hidden_size[-1] if len(self.mlp_hidden_size) else self.embedding_size
        self.device = config['device']
                      
        self.user_num = data.user_num
        self.item_num = data.item_num
        

        user_size_list = [self.embedding_size] + self.mlp_hidden_size
        item_size_list = [self.embedding_size] + self.mlp_hidden_size

        # define layers and loss
        self.user_mlp_layers = MLPLayers(user_size_list, self.dropout_prob, activation='tanh', bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, self.dropout_prob, activation='tanh', bn=True)

        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
         
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        # parameters initialization
        self.apply(self._init_weights)
        self.visual_encoder = load_model(config=config)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def forward(self, input):
        user, item = input
        embed_user = self.user_embedding(user)
        embed_item = self.visual_encoder(item.flatten(0,1))
        user_dnn_out = self.user_mlp_layers(embed_user).unsqueeze(1)
        item_dnn_out = self.item_mlp_layers(embed_item)
        item_dnn_out = item_dnn_out.view(user.shape[0], -1, self.out_size)
        score = (user_dnn_out * item_dnn_out).sum(-1)
        output = score.view(-1,2)
        batch_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
    
    @torch.no_grad()
    def predict(self, user,item_feature):    
        user_feature = self.user_embedding(user)
        user_dnn_out = self.user_mlp_layers(user_feature)        
        scores = torch.matmul(user_dnn_out,item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item(self, item):
        return self.item_mlp_layers(self.visual_encoder(item))

