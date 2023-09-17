import torch
from torch import nn
from REC.model.layers import LightTransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.load import load_model
from REC.model.basemodel import BaseModel

class MOLightSANs(BaseModel):
    input_type = InputType.SEQ
    
    def __init__(self, config, dataload):
        super(MOLightSANs, self).__init__()

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.embedding_size = config['embedding_size']  # same as embedding_size
        self.inner_size = config['inner_size']* self.embedding_size # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.k_interests = config["k_interests"]  
        self.device = config['device']
        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        # define layers and loss
    
        self.visual_encoder = load_model(config=config)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        
        self.trm_encoder = LightTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            k_interests=self.k_interests,
            hidden_size=self.embedding_size,
            seq_len=self.max_seq_length,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
     
        self.position_embedding.weight.data.normal_(mean=0.0, std=self.initializer_range)
        self.trm_encoder.apply(self._init_weights)
        self.LayerNorm.bias.data.zero_()
        self.LayerNorm.weight.data.fill_(1.0)

    def _init_weights(self, module):      
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, items):
  
        batch_size = items.shape[0]
        item_emb = self.visual_encoder(items.flatten(0,1)).view(batch_size,-1,self.embedding_size) #[batch, max_seq_len+2, dim]
 
        input_emb = item_emb[:, :-2, :]      
        target_pos_embs = item_emb[:, -2, :] 
        target_neg_embs = item_emb[:, -1, :] 
        
        
        position_ids = torch.arange(input_emb.size(1), dtype=torch.long, device=self.device)
        position_embedding = self.position_embedding(position_ids)
        
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        output_embs = self.trm_encoder(input_emb, position_embedding, output_all_encoded_layers=False) #[batch, max_seq_len-1, dim]
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

        input_emb = item_feature[item_seq]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        output = self.trm_encoder(input_emb, position_embedding, output_all_encoded_layers=False)
        output_embs = output[-1]
        seq_output = output_embs[:, -1]
               
        scores = torch.matmul(seq_output, item_feature.t())  
        return scores


    @torch.no_grad()
    def compute_item(self, item):
        return self.visual_encoder(item)


