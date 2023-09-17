import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel

class WideDeep(BaseModel):

    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(WideDeep, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.method = config['method'] if config['method'] else 'mean'

        self.item_num = dataload.item_num
        #wide部分
        #相当于wide中Linear变换的w矩阵
        self.wide_item_embedding = nn.Embedding(self.item_num, 1, padding_idx=0)
        self.wide_bias = nn.Parameter(torch.zeros((1,)), requires_grad=True)


        #deep部分
        self.deep_item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)        
        size_list = [self.embedding_size * self.max_seq_length] + self.mlp_hidden_size        
        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        #self.wide_layer = nn.Linear(self.max_seq_length, 1)
        
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

    def forward(self, inputs):  #[batch, 2 , seq_len]
        batch_size = inputs.shape[0]
        #wide_output = self.wide_layer(interaction.float())   #[batch, 2 , seq_len] -> [batch, 2 , 1]
                
        wide_output = torch.sum(self.wide_item_embedding(inputs),dim=-2) + self.wide_bias

        deep_input_emb = self.deep_item_embedding(inputs).view(batch_size*2, -1) ##[batch, 2 , seq_len, dim] -> 
        deep_output = self.mlp_layers(deep_input_emb)
        deep_output = self.deep_predict_layer(deep_output)

    
        output = wide_output.view(-1,2) + deep_output.view(-1,2)
        
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))
        return batch_loss
   

    @torch.no_grad()
    def predict(self, interaction,item_token):
        item_seq, item_len = interaction    
        batch_size = item_seq.shape[0]   
        item_seq = item_seq.flatten(0,1)   
        wide_output = torch.sum(self.wide_item_embedding(item_seq), dim=-2) + self.wide_bias
        deep_input_emb = self.deep_item_embedding(item_seq).view(item_seq.shape[0],-1)
        deep_output = self.mlp_layers(deep_input_emb)
        deep_output = self.deep_predict_layer(deep_output)  

    
        output = wide_output + deep_output
        scores = output.view(batch_size, -1)
        return scores

    @torch.no_grad()   
    def compute_item_all(self):
        return None
 
        #return torch.arange(0,self.n_items).to(self.device)
