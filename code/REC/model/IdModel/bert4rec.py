import torch
from torch import nn
from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel

class BERT4Rec(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(BERT4Rec, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  # same as embedding_size
        self.inner_size = self.hidden_size*config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.mask_ratio = config['mask_ratio']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.initializer_range = config['initializer_range']
        self.device = config['device']
        # load dataset info
        self.item_num = dataload.item_num
        
        self.mask_token = self.item_num
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num+1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
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



    def reconstruct_test_data(self, item_seq):

        padding = torch.full(size=(item_seq.size(0),1), fill_value=self.mask_token, dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding), dim=-1)  

        return item_seq



   
    def forward(self, input):
        #masked_sequence, instance, neg_items, masked_index
        items, masked_index = input
        input_ids = items[:, 0]
        item_emb = self.item_embedding(items) #[batch, 3, max_seq_len+1, dim]
        input_items_embs = item_emb[:, 0]  #[batch, max_seq_len+1, dim]
        pos_items_embs = item_emb[:, 1]   #[batch, max_seq_len+1, dim]
        neg_items_embs = item_emb[:, 2]

        position_ids = torch.arange(end=input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_items_embs + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(input_ids)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
        output_embs = output_embs[-1]
                
        indices = torch.where(masked_index != 0)
        batch = masked_index.shape[0]
        seq_output = output_embs[indices]
        pos_items_emb = pos_items_embs[indices]
        neg_items_emb = neg_items_embs[indices]


        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B*mask_len]
        neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B*mask_len]

        loss = - (torch.log(1e-8 + torch.sigmoid(pos_score - neg_score))).sum(-1) 
   
        return loss/batch
       
    @torch.no_grad()
    def predict(self, item_seq, item_feature):     
        item_seq = self.reconstruct_test_data(item_seq)
     
        input_items_embs = self.item_embedding(item_seq)  #[batch, max_seq_len+1, dim]
     
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_items_embs + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
      
        seq_output = output_embs[-1][:, -1]
        scores = torch.matmul(seq_output, item_feature.t())  
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight[:self.item_num]

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask