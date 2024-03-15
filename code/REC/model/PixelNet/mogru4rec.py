import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.load import load_model
from REC.model.basemodel import BaseModel

class MOGRU4Rec(BaseModel):
    input_type = InputType.SEQ
    
    def __init__(self, config, dataload):
        super(MOGRU4Rec, self).__init__()

        # load parameters info

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size'] * config['embedding_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        # define layers and loss
    
        self.visual_encoder = load_model(config=config)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        
        xavier_uniform_(self.gru_layers.weight_hh_l0)
        xavier_uniform_(self.gru_layers.weight_ih_l0)
        xavier_normal_(self.dense.weight)


    def forward(self, interaction):
        items, masked_index = interaction   
        batch_size = masked_index.shape[0]
        item_emb = self.visual_encoder(items.flatten(0,1)).view(batch_size, -1, 2, self.embedding_size) #[batch, 2, max_seq_len+1, dim]
        pos_items_embs = item_emb[:, :, 0]  
        neg_items_embs = item_emb[:, :, 1]   

        input_emb = pos_items_embs[:, :-1, :]       
        target_pos_embs = pos_items_embs[:, 1:, :]  
        target_neg_embs = neg_items_embs[:, 1:, :] 
                
        input_emb_dropout = self.emb_dropout(input_emb)
        gru_output, _ = self.gru_layers(input_emb_dropout)
        gru_output = self.dense(gru_output)

        pos_score = (gru_output * target_pos_embs).sum(-1)   
        neg_score = (gru_output * target_neg_embs).sum(-1)  
        
        loss = - (torch.log((pos_score - neg_score).sigmoid() + 1e-8)*masked_index).sum(-1) 
        return loss.mean(-1)

    @torch.no_grad()
    def predict(self, item_seq, item_feature):
       
        item_emb = item_feature[item_seq]

        item_seq_emb_dropout = self.emb_dropout(item_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)       
        hidden = gru_output[:, -1]                     
        scores = torch.matmul(hidden, item_feature.t())  
        return scores


    @torch.no_grad()
    def compute_item(self, item):
        return self.visual_encoder(item)


