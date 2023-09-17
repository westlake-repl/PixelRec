import torch
import torch.nn as nn
from REC.utils import InputType
from REC.model.basemodel import BaseModel
import torch.nn.functional as F
import math
import numpy as np
#torch.set_default_dtype(torch.float64)
class SRGNN(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, data):
        super(SRGNN, self).__init__()
        self.hidden_size = config['embedding_size']  
        self.step = config['step'] 

        self.device = config['device']
        self.item_num = data.item_num
        
        self.embedding = nn.Embedding(self.item_num, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=self.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
          
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self._reset_parameters()
    
    
    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
   
 
    
    def seq_modeling(self, alias_inputs, A, hidden, mask):
        gnn_output = self.gnn(A, hidden)
        seq_hidden = []
        for i in range(len(alias_inputs)):
            seq_hidden.append(gnn_output[i][alias_inputs[i]])
        seq_hidden = torch.stack(seq_hidden)
        
        ht = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size 取的最后一位作为global
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.shape[0], -1, 1).float(), 1)    
        a = self.linear_transform(torch.cat([a, ht], 1))
        return a
    
    def forward(self, input):     
        alias_inputs, A, items, mask, targets = input
        hidden = self.embedding(items)
        seq_output = self.seq_modeling(alias_inputs, A, hidden, mask).unsqueeze(1)  #[batch,1, dim]
        target_output =self.embedding(targets)  #[batch,2, dim]
        score = (seq_output * target_output).sum(-1)
        output = score.view(-1,2)
        batch_loss = -torch.mean(1e-8+torch.log(torch.sigmoid(torch.matmul(output, self.weight))))        
        return batch_loss
      
   
    @torch.no_grad()
    def predict(self, input,item_feature):
        alias_inputs, A, items, mask = input
        hidden = item_feature[items]
        seq_output = self.seq_modeling(alias_inputs, A, hidden, mask)       
        scores = torch.matmul(seq_output,item_feature.t())
        return scores

    @torch.no_grad()    
    def compute_item_all(self):
        embed_item = self.embedding.weight
        return embed_item



class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden