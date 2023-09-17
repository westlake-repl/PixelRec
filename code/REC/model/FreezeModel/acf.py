import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F

class ACF(BaseModel):
    input_type = InputType.SEQ  
    def __init__(self, config, dataload):
        super(ACF, self).__init__()
        self.embedding_size = config['embedding_size']    
        self.device = config['device']
              
        self.user_num = dataload.user_num
        self.item_num = dataload.item_num
        
        self.v_feat_path = config['v_feat_path']   
        v_feat = np.load(self.v_feat_path, allow_pickle=True)   
        self.v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        self.feature_dim = self.v_feat.shape[-1]
        
        self.item_model = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.user_model = ACFUserNet(
                            num_users=self.user_num,
                            num_items=self.item_num,
                            emb_dim=self.embedding_size,
                            input_feature_dim=self.feature_dim,
                            profile_embedding=self.item_model,
                            device=self.device)

        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self._kaiming_(self.item_model)

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


    def forward(self, inputs):
        user_id = inputs[:, -1]
        profile_ids = inputs[:, :-3]
        items = inputs[:, -3:-1]
        item_embs = self.item_model(items)

        profile_features = self.v_feat[profile_ids]
        profile_mask = (profile_ids != 0)
        user_output = self.user_model(user_id, profile_ids, profile_features, profile_mask)
        user_embs = user_output['user'].unsqueeze(1)
        
        score = (user_embs * item_embs).sum(-1)
        output = score.view(-1,2)    
        batch_loss = -torch.mean(torch.log(1e-8+torch.sigmoid(torch.matmul(output, self.weight))))        
        return batch_loss


    @torch.no_grad()
    def predict(self, inputs, item_feature):
        user_id = inputs[:, -1]
        profile_ids = inputs[:, :-1]
        
        profile_features = self.v_feat[profile_ids]
        profile_mask = (profile_ids != 0)
        user_output = self.user_model(user_id, profile_ids, profile_features, profile_mask)
        user_embs = user_output['user']
                      
        scores = torch.matmul(user_embs, item_feature.t())  
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_model.weight



class ACFUserNet(nn.Module):
    """
    Get user embedding accounting to surpassed items
    """

    def __init__(self, num_users, num_items, emb_dim=128, input_feature_dim=0, profile_embedding=None, device=None):
        super().__init__()
        self.pad_token = 0
        self.emb_dim = emb_dim
        reduced_feature_dim = emb_dim
        self.feats = ACFFeatureNet(emb_dim, input_feature_dim, reduced_feature_dim) if input_feature_dim > 0 else None

        self.user_embedding = nn.Embedding(num_users, emb_dim)

        self.profile_embedding = profile_embedding

        self.w_u = nn.Linear(emb_dim, emb_dim)
        self.w_p = nn.Linear(emb_dim, emb_dim)
        self.w_x = nn.Linear(emb_dim, emb_dim)
        self.w = nn.Linear(emb_dim, 1)

        self._kaiming_(self.user_embedding)
        self._kaiming_(self.w_u)
        self._kaiming_(self.w_p)
        self._kaiming_(self.w_x)
        self._kaiming_(self.w)

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.device = device

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def forward(self, user_ids, profile_ids, features, profile_mask, return_component_attentions=False,
                return_profile_attentions=False, return_attentions=False):
        return_component_attentions = return_component_attentions or return_attentions
        return_profile_attentions = return_profile_attentions or return_attentions

        batch_size = user_ids.nelement()
        user = self.user_embedding(user_ids)

        if profile_ids.nelement() != 0:
            profile = self.profile_embedding(profile_ids)
        else:
            profile = torch.zeros((batch_size, 0, self.emb_dim), device=self.device)

        if self.feats is not None:
            features = features.flatten(start_dim=2, end_dim=3) # Add #[B, 7，7, D]-》[B, 49, D]
            feat_output = self.feats(user, features, profile_mask, return_attentions=return_component_attentions)
            components = feat_output['pooled_features']  #[B, P, D]
        else:
            components = torch.tensor([], device=self.device)

        user = self.w_u(user)    #[B, D]
        profile_query = self.w_p(profile)   #[B, P, D]
        components = self.w_x(components)   #[B, P, D]

        profile_query = profile_query.permute((1,0,2))  #[P, B, D]
        components = components.permute((1,0,2))   #[P, B, D]

        alpha = F.relu(user + profile_query + components) 
        alpha = self.w(alpha)  

        profile_mask = profile_mask.permute((1,0))
        profile_mask = profile_mask.unsqueeze(-1)
        alpha = alpha.masked_fill(torch.logical_not(profile_mask), float('-inf'))
        alpha = F.softmax(alpha, dim=0)

        is_nan = torch.isnan(alpha)
        if is_nan.any():
            # softmax is nan when all elements in dim 0 are -infinity or infinity
            alpha = alpha.masked_fill(is_nan, 0.0)

        alpha = alpha.permute((1,0,2))
        user_profile = (alpha * profile).sum(dim=1)

        user = user + user_profile
        output = {'user': user}
        if return_component_attentions:
            output['component_attentions'] = feat_output['attentions']
        if return_profile_attentions:
            output['profile_attentions'] = alpha.squeeze(-1)

        return output


class ACFFeatureNet(nn.Module):
    """
    Process auxiliary item features into latent space.
    All items for user can be processed in batch.
    """
    def __init__(self, emb_dim, input_feature_dim, feature_dim, hidden_dim=None, output_dim=None):
        super().__init__()

        if not hidden_dim:
            hidden_dim = emb_dim

        if not output_dim:
            output_dim = emb_dim

        # e.g. 2048 => 512
        self.dim_reductor = nn.Linear(input_feature_dim, feature_dim)

        self.w_x = nn.Linear(feature_dim, hidden_dim)
        self.w_u = nn.Linear(emb_dim, hidden_dim)

        self.w = nn.Linear(hidden_dim, 1)

        self._kaiming_(self.dim_reductor)
        self._kaiming_(self.w_x)
        self._kaiming_(self.w_u)
        self._kaiming_(self.w)

    def _kaiming_(self, layer):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def forward(self, user, components, profile_mask, return_attentions=False):
        x = self.dim_reductor(components) # Add
        x = F.relu(x)
        x = x.movedim(0, -2) # BxPxHxD => PxHxBxD
        #B是batch_size   P是profile_len  H是height即7x7里的49   D是dimension 2048
        x_tilde = self.w_x(x)
        user = self.w_u(user)

        beta = F.relu(x_tilde + user)
        beta = self.w(beta)

        beta = F.softmax(beta, dim=1)

        x = (beta * x).sum(dim=1)
        x = x.movedim(-2, 0) # PxBxD => BxPxD

        feature_dim = x.shape[-1]
        profile_mask = profile_mask
        profile_mask = profile_mask.unsqueeze(-1).expand((*profile_mask.shape, feature_dim))

        x = profile_mask * x
        output = {'pooled_features': x}
        if return_attentions:
            output['attentions'] = beta.squeeze(-1).squeeze(-1)
        return output