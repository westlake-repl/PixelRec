import torch
from torch import nn
from REC.model.layers import Identity
from REC.utils.enum_type import InputType
from REC.model.load import load_model
from REC.model.basemodel import BaseModel
import torch.functional as F
import torch.nn.functional as F2
from torch.nn.init import xavier_normal_, constant_, uniform_
import numpy as np

class MONextItNet(BaseModel):
    input_type = InputType.SEQ    
    def __init__(self, config, dataload):
        super(MONextItNet, self).__init__()

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.residual_channels = config['embedding_size']
        self.block_num = config['block_num']
        self.dilations = config['dilations'] * self.block_num
        self.kernel_size = config['kernel_size']
        self.item_num = dataload.item_num
        # define layers and loss
        self.visual_encoder = load_model(config=config)

        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        if config['final_layer']:
            self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)  #这一层到底有没有用啊？？？
        else:
            self.final_layer = Identity()
        
        self.residual_blocks.apply(self._init_weights)
        self.final_layer.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.item_num)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)


    def forward(self, interaction):
        items, masked_index = interaction   
        batch_size = masked_index.shape[0]
        item_emb = self.visual_encoder(items.flatten(0,1)).view(batch_size, -1, 2, self.embedding_size) #[batch, 2, max_seq_len+1, dim]
        pos_items_embs = item_emb[:, :, 0]  
        neg_items_embs = item_emb[:, :, 1]  

        input_emb = pos_items_embs[:, :-1, :]       
        target_pos_embs = pos_items_embs[:, 1:, :] 
        target_neg_embs = neg_items_embs[:, 1:, :] 
                
        dilate_outputs = self.residual_blocks(input_emb)        
        dilate_outputs = self.final_layer(dilate_outputs)
        pos_score = (dilate_outputs * target_pos_embs).sum(-1)   
        neg_score = (dilate_outputs * target_neg_embs).sum(-1)   
        
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)*masked_index).sum(-1) 
        return loss.mean(-1)

    @torch.no_grad()
    def predict(self, item_seq, item_feature):
        item_emb = item_feature[item_seq]
        dilate_outputs = self.residual_blocks(item_emb)        
        dilate_outputs = self.final_layer(dilate_outputs)
        hidden = dilate_outputs[:, -1]                           
        scores = torch.matmul(hidden, item_feature.t())  # [B n_items]
        return scores


    @torch.no_grad()
    def compute_item(self, item):
        return self.visual_encoder(item)

   





class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F2.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F2.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad
