from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import lmdb
import pickle
import random
import math
import os

# Image_Mean = [0.4860599,  0.4426124,  0.43379018]
# Image_Std = [0.31636897, 0.3010678,  0.30478135]

Image_Mean = [0.5,  0.5,  0.5]
Image_Std = [0.5, 0.5,  0.5]
Resize = 224

#数据形式为 [[user_seq], [neg_item_seq]] , [mask]
class SEQTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']        

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):        
        masked_index = []              
        neg_item = []
        item_seq_len = len(item_seq)           
        for i in range(item_seq_len -1):        
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length)
        neg_item = self._padding_sequence(neg_item, self.max_seq_length)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)        
        return item_seq, neg_item, masked_index    
    
    def __getitem__(self, index): 
        #最长长度为maxlen+1, 及若max_len是5
        #则存在    1,2,3,4,5,6序列,
        #pos       2,3,4,5,6
        #neg       0,8,9,7,9,8
        #mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]      
        item_seq, neg_item, masked_index  = self.reconstruct_train_data(item_seq)
        items = torch.stack((item_seq,neg_item))
     
        return items, masked_index 



class MOSEQTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config
    
        
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq) 
        self.id2token = dataload.id2token['item_id']       
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']  

        self.image_path = config['image_path']
        self.load_content()
                         
    def __len__(self):
        return self.length
    
    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            transforms.Resize((Resize,Resize)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,Resize,Resize))
        

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):        
        masked_index = []              
        neg_item = []
        item_seq_len = len(item_seq)           
        for i in range(item_seq_len -1):        
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length)
        neg_item = self._padding_sequence(neg_item, self.max_seq_length)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)        
        return item_seq, neg_item, masked_index    
    
    def __getitem__(self, index): 

        item_seq = self.train_seq[index]      
        item_seq, neg_item, masked_index  = self.reconstruct_train_data(item_seq)
        item_seq_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_item] 
        PAD_token = self.id2token[0]
        items_modal = []
    
        with self.env.begin() as txn:
            for (item, neg)  in zip(item_seq_token, neg_items_token):
                if item == PAD_token:
                    item_i = self.pad_image                    
                else:                        
                    byteflow_i = txn.get(item.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                items_modal.append(item_i) 
                                                
                if neg == PAD_token :
                    item_i = self.pad_image                    
                else:
                    byteflow_i = txn.get(neg.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB'))
                items_modal.append(item_i)    

        items_modal = torch.stack(items_modal)  #[max_len*2, 3, 224, 224]  
    
        return items_modal, masked_index  


#数据形式为 [user_id] , [pos_item, neg_item]
class PairTrainDataset(Dataset):
    def __init__(self,config,dataload):    
        self.dataload = dataload        
        self.user_seq = dataload.user_seq
        self.item_num = dataload.item_num 
        self.train_uid = dataload.train_feat['user_id']
        self.train_iid = dataload.train_feat['item_id']
        self.length = len(self.train_uid)        
        
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
   
    def __getitem__(self, index): 
        user = self.train_uid[index]
        item_i = self.train_iid[index]

        used = self.user_seq[user][:-2]
        item_j = random.randint(1, self.item_num-1)
        while item_j in used:
            item_j = random.randint(1, self.item_num-1)
        
        item = torch.tensor([item_i,  item_j])
        user = torch.tensor(user)
        return user, item
 


class MOPairTrainDataset(Dataset):
    def __init__(self,config,dataload):    
        self.dataload = dataload        
        self.user_seq = dataload.user_seq
        self.item_num = dataload.item_num 
        self.train_uid = dataload.train_feat['user_id']
        self.train_iid = dataload.train_feat['item_id']        
        self.id2token = dataload.id2token['item_id']
        self.length = len(self.train_uid)        

        self.device = config['device']                  
        self.image_path = config['image_path']
        self.load_content()
                         
    def __len__(self):
        return self.length
    
    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         
        ])
        self.pad_image = torch.zeros((3,224,224))
    
   
    def __getitem__(self, index): 
        user = self.train_uid[index]
        item_i = self.train_iid[index]

        used = self.user_seq[user][:-2]
        item_j = random.randint(1, self.item_num-1)
        while item_j in used:
            item_j = random.randint(1, self.item_num-1)
        
        item_token_i = self.id2token[item_i]
        item_token_j = self.id2token[item_j]
                     
        with self.env.begin() as txn:
            byteflow_i = txn.get(item_token_i.encode('ascii'))
            IMAGE_i = pickle.loads(byteflow_i)
            item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')).unsqueeze(0)
            byteflow_j = txn.get(item_token_j.encode('ascii'))
            IMAGE_j = pickle.loads(byteflow_j)
            item_j = self.feature_extractor(Image.fromarray(IMAGE_j.get_image()).convert('RGB')).unsqueeze(0)
            item = torch.cat((item_i, item_j))
        return user, item


#数据形式为 [user_seq, pos_item, neg_item]
class TwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+2
        self.device = config['device']         

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index): 
        item_seq = list(self.train_seq[index])  
        neg_item = self._neg_sample(item_seq)
        item_seq +=[neg_item]
        items = self._padding_sequence(item_seq, self.max_seq_length)     
        return items  



class SampleTwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.iter_num = dataload.inter_num
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+2
        self.device = config['device']         

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index):
        train_seq = list(self.train_seq[index])
        items = []
        for idx, item in enumerate(train_seq):
            neg_item = self._neg_sample(train_seq)
            item_list = train_seq[:idx] + train_seq[idx+1:] + [item] + [neg_item]
            items_pad = self._padding_sequence(item_list, self.max_seq_length)
            items.append(items_pad) 
        return torch.stack(items)  



#数据形式为 [[pos_user_seq], [neg_user_seq]]
class OneTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index): 
        item_seq = list(self.train_seq[index])
        item_seq = self._padding_sequence(item_seq, self.max_seq_length)
        neg_item = item_seq.clone()
        neg_item[-1] = self._neg_sample(item_seq)
        items = torch.stack((item_seq,neg_item))     
        return items


#数据形式为 [[pos_user_seq], [neg_user_seq]]
class SampleOneTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index):
        train_seq = list(self.train_seq[index])
        items = []
        for idx, item in enumerate(train_seq):
            neg_item = self._neg_sample(train_seq)
            item_list = train_seq[:idx] + train_seq[idx+1:] + [item] 
            pos_pad = self._padding_sequence(item_list, self.max_seq_length)
            neg_pad = pos_pad.clone()
            neg_pad[-1] = neg_item
            items_pad = torch.stack((pos_pad,neg_pad))  
            items.append(items_pad) 
         
        return torch.stack(items)




class BERT4RecTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config
        
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq']         
        self.length = len(self.train_seq)
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1   
        self.mask_ratio = config['mask_ratio']
        self.device = config['device']        
        self.mask_token = self.item_num  #最后一位的index, 即原本的最后一位index是n_item-1, mask的index是n_item         
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)       

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence):
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-self.max_seq_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq): 
        #item_seq 长度： max_len + 1
        neg_item = []        
        masked_sequence = [] 
        masked_index = []     
        
        for index_id, item in enumerate(item_seq):
            prob = random.random()
            if prob < self.mask_ratio:            
                neg_item.append(self._neg_sample(item_seq))
                masked_sequence.append(self.mask_token)
                masked_index.append(1)
            else :
                neg_item.append(0)
                masked_sequence.append(item)
                masked_index.append(0)

        item_seq = self._padding_sequence(list(item_seq))        
        neg_items = self._padding_sequence(neg_item)
        masked_sequence = self._padding_sequence(masked_sequence)
        masked_index = self._padding_sequence(masked_index)
        return item_seq, neg_items, masked_sequence, masked_index       
    
    
    def __getitem__(self, index): 

        item_seq = self.train_seq[index]      
        item_seq, neg_items, masked_sequence, masked_index  = self.reconstruct_train_data(item_seq)        
        items = torch.stack((masked_sequence, item_seq, neg_items), dim=0)    
        return items, masked_index 


class MOTwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        self.id2token = dataload.id2token['item_id']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+2
        self.device = config['device']         
        self.image_path = config['image_path']
        self.load_content()
                         
    def __len__(self):
        return self.length
    
    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,224,224))
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index): 
        item_seq = list(self.train_seq[index])  
        neg_item = self._neg_sample(item_seq)
        item_seq +=[neg_item]
        items = self._padding_sequence(item_seq, self.max_seq_length)
        item_seq_token = self.id2token[items]
        PAD_token = self.id2token[0]
        items_modal = []
        #pos_neg_modal = np.zeros((2, self.mask_item_length, 3, self.image_size, self.image_size))    
        with self.env.begin() as txn:
            for item in item_seq_token:
                if item == PAD_token:
                    item_i = self.pad_image                    
                else:                        
                    byteflow_i = txn.get(item.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                items_modal.append(item_i) 
                                                
        items_modal = torch.stack(items_modal)  #[max_len, 3, 224, 224]  
    
        return items_modal#, items       



class BaseDataset(Dataset):
    def __init__(self,config,dataload):
        pass
      

           
    def __len__(self):
        return 0
    




class ACFTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq']
        self.user_id = dataload.train_feat['user_id']
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+3 #[pos, neg , uid]
        self.device = config['device']         

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index):
        user_id = self.user_id[index] 
        item_seq= list(self.train_seq[index])  
        neg_item = self._neg_sample(item_seq)
        item_seq +=[neg_item, user_id]    
        items = self._padding_sequence(item_seq, self.max_seq_length)     
        return items 



class SampleACFTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        train_seq_before = dataload.train_feat['item_seq']
        user_id_before = dataload.train_feat['user_id']
                
        train_seq_after = []
        user_id_after = []
        for (uid, train_seq) in zip(user_id_before, train_seq_before):
            train_seq = list(train_seq)
            for idx, item in enumerate(train_seq):
                item_list = train_seq[:idx] + train_seq[idx+1:] + [item] 
                train_seq_after.append(item_list)
                user_id_after.append(uid)

        self.train_seq = train_seq_after
        self.user_id = user_id_after
        self.length = len(self.train_seq)
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+3 #[pos, neg , uid]
        self.device = config['device']         

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index):
        user_id = self.user_id[index] 
        train_seq= self.train_seq[index] 

        neg_item = self._neg_sample(train_seq)
        item_list = train_seq + [neg_item,user_id]
        items_pad = self._padding_sequence(item_list, self.max_seq_length)
        return items_pad 



class MOSampleOneTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.id2token = dataload.id2token['item_id']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1  #+target
        self.device = config['device']       
        
        self.image_path = config['image_path']
        self.load_content()
           
    def __len__(self):
        return self.length

    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,224,224))
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index):
        train_seq = list(self.train_seq[index])
        seq_len = len(train_seq)
        all_item = [0] + train_seq[:]
        for _ in range(seq_len):
            neg_item = self._neg_sample(train_seq)
            all_item.append(neg_item)
        all_item_token = self.id2token[all_item]
        PAD_token = self.id2token[0]
        all_item_modal = []
        with self.env.begin() as txn:
            for item in all_item_token:
                if item == PAD_token:
                    item_i = self.pad_image                    
                else:                        
                    byteflow_i = txn.get(item.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                all_item_modal.append(item_i) 
                                                

        items_index = []
        all_item_index = [idx for idx, _ in enumerate(all_item)]
        pos_item_index = all_item_index[1:seq_len+1]
        neg_item_index = all_item_index[seq_len+1:]
        for idx, pos_index in enumerate(pos_item_index):
            neg_index = neg_item_index[idx]
            item_list = pos_item_index[:idx] + pos_item_index[idx+1:] + [pos_index]
            pos_pad = self._padding_sequence(item_list, self.max_seq_length)
            neg_pad = pos_pad.clone()
            neg_pad[-1] = neg_index
            items_pad = torch.stack((pos_pad,neg_pad))  
            items_index.append(items_pad)  
        
        mask = pos_pad != 0
        mask = mask.long()
        return torch.stack(items_index), mask, torch.stack(all_item_modal)


class MOSampleTwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.id2token = dataload.id2token['item_id']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+2  #+target
        self.device = config['device']       
        
        self.image_path = config['image_path']
        self.load_content()
           
    def __len__(self):
        return self.length

    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            #transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,224,224))
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index):
        train_seq = list(self.train_seq[index])
        seq_len = len(train_seq)
        all_item = [0] + train_seq[:]
        for _ in range(seq_len):
            neg_item = self._neg_sample(train_seq)
            all_item.append(neg_item)
        all_item_token = self.id2token[all_item]
        PAD_token = self.id2token[0]
        all_item_modal = []
        with self.env.begin() as txn:
            for item in all_item_token:
                if item == PAD_token:
                    item_i = self.pad_image                    
                else:                        
                    byteflow_i = txn.get(item.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                all_item_modal.append(item_i) 
        
        
        items_index = []
        all_item_index = [idx for idx, _ in enumerate(all_item)]
        pos_item_index = all_item_index[1:seq_len+1]
        neg_item_index = all_item_index[seq_len+1:]
        for idx, pos_index in enumerate(pos_item_index):
            neg_index = neg_item_index[idx]
            item_list = pos_item_index[:idx] + pos_item_index[idx+1:] + [pos_index] +[neg_index]
            items_pad = self._padding_sequence(item_list, self.max_seq_length)
            items_index.append(items_pad) 
        
        mask = items_pad != 0
        mask = mask.long()
        return torch.stack(items_index), mask, torch.stack(all_item_modal)


class MOBERT4RecTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config
        
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)      
        self.id2token = list(dataload.id2token['item_id'])
        self.id2token.append('mask')
        self.id2token = np.array(self.id2token)
        
        self.image_path = config['image_path']
        self.load_content()
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1   
        self.mask_ratio = config['mask_ratio']
        self.device = config['device']        
        self.mask_token = self.item_num  #最后一位的index, 即原本的最后一位index是n_item-1, mask的index是n_item         
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)       

    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            transforms.Resize((Resize,Resize)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,Resize,Resize))
        self.mask_image = torch.ones((3,Resize,Resize))
    

    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence):
        pad_len = self.max_seq_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-self.max_seq_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq): 
        #item_seq 长度： max_len + 1
        neg_item = []        
        masked_sequence = [] 
        masked_index = []     
        
        for index_id, item in enumerate(item_seq):
            prob = random.random()
            if prob < self.mask_ratio:            
                neg_item.append(self._neg_sample(item_seq))
                masked_sequence.append(self.mask_token)
                masked_index.append(1)
            else :
                neg_item.append(0)
                masked_sequence.append(item)
                masked_index.append(0)

        item_seq = self._padding_sequence(list(item_seq))        
        neg_items = self._padding_sequence(neg_item)
        masked_sequence = self._padding_sequence(masked_sequence)
        masked_index = self._padding_sequence(masked_index)
        return item_seq, neg_items, masked_sequence, masked_index       
    
    
    def __getitem__(self, index): 

        item_seq = self.train_seq[index]      
        item_seq, neg_items, masked_sequence, masked_index  = self.reconstruct_train_data(item_seq)   
        
        item_seq_token = self.id2token[masked_sequence]
        pos_items_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_items] 

        PAD_token = self.id2token[0]
        masked_pos_neg_modal = []
        #pos_neg_modal = np.zeros((2, self.mask_item_length, 3, self.image_size, self.image_size))    
        with self.env.begin() as txn:

            for (item, pos, neg)  in zip(item_seq_token, pos_items_token, neg_items_token):
                if item == 'mask' or item == PAD_token:
                    item_i = self.mask_image                    
                else:                        
                    byteflow_i = txn.get(item.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                masked_pos_neg_modal.append(item_i) 
                
                if pos == PAD_token :
                    item_i = self.pad_image                    
                else:
                    byteflow_i = txn.get(pos.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB'))
                masked_pos_neg_modal.append(item_i)      #[mask_len, 3, 224, 224]
            
            
                if neg == PAD_token :
                    item_i = self.pad_image                   
                else:
                    byteflow_i = txn.get(neg.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB'))
                masked_pos_neg_modal.append(item_i)    #[mask_len, 3, 224, 224]



        masked_pos_neg_modal = torch.stack(masked_pos_neg_modal)  #[mask_len*3, 3, 224, 224]  [item, pos,neg,item, pos,neg...]排列        
        return masked_sequence, masked_pos_neg_modal, masked_index 



class GraphTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.device = config['device']        

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return np.array(sequence, dtype=np.int)

    
    def __getitem__(self, index): 
        train_seq = list(self.train_seq[index]) 
        item_seq = train_seq[:-1]
        neg_item = self._neg_sample(train_seq)  
        target = [train_seq[-1], neg_item]
        item_length = min(len(item_seq), self.max_seq_length)
        masked_index = [1]*item_length
        item_seq = self._padding_sequence(item_seq, self.max_seq_length)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length)  
        target = np.array(target)
     
        return item_seq,masked_index,target


class MOGraphTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.id2token = dataload.id2token['item_id']
        
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.device = config['device']        
        
        self.image_path = config['image_path']
        self.load_content()
           
    def __len__(self):
        return self.length
    
    
    def load_content(self):
        self.env = lmdb.open(self.image_path, subdir=os.path.isdir(self.image_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.feature_extractor = transforms.Compose([
            transforms.Resize((Resize,Resize)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
            transforms.Normalize(mean=Image_Mean, std=Image_Std )         

        ])
        self.pad_image = torch.zeros((3,Resize,Resize))


    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return np.array(sequence, dtype=np.int)

    
    def __getitem__(self, index): 
        train_seq = list(self.train_seq[index]) 
        input_seq = train_seq[:-1]
        neg_item = self._neg_sample(train_seq)  
        target = [train_seq[-1], neg_item]
        item_length = min(len(input_seq), self.max_seq_length)
        masked_index = [1]*item_length
        #item_seq = self._padding_sequence(item_seq, self.max_seq_length+2)
        input_seq = self._padding_sequence(input_seq, self.max_seq_length)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length)  
        node = np.unique(input_seq)
        input_items =node.tolist()     #+ (max_n_node - len(node)) * [0]
        #items_all = input_items + target
        input_token_seq = self.id2token[input_items]
        target_token_seq = self.id2token[target]
        PAD_token = self.id2token[0]        
        input_modal_seq = []
        target_modal_seq = []
        with self.env.begin() as txn:
            for item_token in input_token_seq:
                if item_token == PAD_token:
                    item_i = self.pad_image                    
                else:                        
                    byteflow_i = txn.get(item_token.encode('ascii'))
                    IMAGE_i = pickle.loads(byteflow_i)                        
                    item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                input_modal_seq.append(item_i) 
            
            for item_token in target_token_seq:                     
                byteflow_i = txn.get(item_token.encode('ascii'))
                IMAGE_i = pickle.loads(byteflow_i)                        
                item_i = self.feature_extractor(Image.fromarray(IMAGE_i.get_image()).convert('RGB')) 
                target_modal_seq.append(item_i) 
        
        return input_seq, masked_index, input_modal_seq, target_modal_seq  #[seq_len], [seq_len], 
