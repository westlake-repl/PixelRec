import torch
from torch import nn

from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.load import load_weights
from REC.utils import set_color
import numpy as np

class FSASRec(nn.Module):
    input_type = InputType.SEQ
    
    def __init__(self, config, dataload):
        super(FSASRec, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  
        self.inner_size = config['inner_size']  
        
        self.inner_size *= self.hidden_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        self.config = config
        # define layers and loss
        config['item_num'] = dataload.item_num                   
        self.item_embedding = load_weights(config)
        

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
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
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



    def forward(self, interaction):
        items, masked_index = interaction   
        item_emb = self.item_embedding(items) 
        pos_items_embs = item_emb[:, 0, :]  
        neg_items_embs = item_emb[:, 1, :]   
        input_emb = pos_items_embs[:, :-1, :]       
        target_pos_embs = pos_items_embs[:, 1:, :]  
        target_neg_embs = neg_items_embs[:, 1:, :] 
        
        
        position_ids = torch.arange(masked_index.size(1), dtype=torch.long, device=masked_index.device)
        position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(masked_index,bidirectional=False)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
        output_embs = output_embs[-1]
        pos_score = (output_embs * target_pos_embs).sum(-1)   
        neg_score = (output_embs * target_neg_embs).sum(-1)   
        
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)*masked_index).sum(-1) 
        return loss.mean(-1)

    @torch.no_grad()
    def predict(self, item_seq, item_feature):
       
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq,bidirectional=False)

        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)
        output_embs = output[-1]
        seq_output = output_embs[:, -1]
               
        scores = torch.matmul(seq_output, item_feature.t())  
        
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        if self.config['semantic_model']:
            return self.item_embedding.pq_code_embedding(self.item_embedding.pq_codes).mean(dim=-2)
        elif self.config['freeze_model']:
            return self.item_embedding.rec_fc(self.item_embedding.item_weights)
        elif self.config['hybrid_model']:
            return self.item_embedding.rec_fc(torch.cat((self.item_embedding.item_weights, self.item_embedding.item_id_embedding.weight), dim=-1))
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'
