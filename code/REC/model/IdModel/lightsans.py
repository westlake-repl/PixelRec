import torch
from torch import nn

from REC.model.layers import LightTransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel


class LightSANs(BaseModel):
    input_type = InputType.SEQ
    
    def __init__(self, config, dataload):
        super(LightSANs, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  
        self.inner_size = config['inner_size']  

        self.k_interests = config["k_interests"]    
        self.inner_size *= self.hidden_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.device = config['device']

        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.hidden_size,
            seq_len=self.max_seq_length,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
      

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    
    def forward(self, interaction):
        item_emb = self.item_embedding(interaction)
 
        input_emb = item_emb[:, :-2, :]       
        target_pos_embs = item_emb[:, -2, :]  
        target_neg_embs = item_emb[:, -1, :] 
        
        
        position_ids = torch.arange(input_emb.size(1), dtype=torch.long, device=self.device)
        position_embedding = self.position_embedding(position_ids)
        
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        output_embs = self.trm_encoder(input_emb, position_embedding, output_all_encoded_layers=False) 
        output_embs = output_embs[-1]
        output_embs = output_embs[:, -1, :]
        
        pos_score = (output_embs * target_pos_embs).sum(-1)   
        neg_score = (output_embs * target_neg_embs).sum(-1)   

        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8))
        return loss.mean(-1)

   
    @torch.no_grad()
    def predict(self, item_seq, item_feature):
       
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_embedding = self.position_embedding(position_ids)

        input_emb = self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        output = self.trm_encoder(input_emb, position_embedding, output_all_encoded_layers=False)
        output_embs = output[-1]
        seq_output = output_embs[:, -1]
               
        scores = torch.matmul(seq_output, item_feature.t())  
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight


