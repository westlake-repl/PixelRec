import copy
import importlib
import os
import pickle
from logging import getLogger
from REC.data.dataset import *
from REC.utils import set_color
from functools import partial
from .data import Data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
def load_data(config):
    dataload = Data(config)
    return dataload


def bulid_dataloader(config,dataload):
    '''
    split dataset, generate user history sequence, train/valid/test dataset
    '''
    dataset_dict = {
        'BERT4Rec': ('BERT4RecTrainDataset', 'SeqEvalDataset','seq_eval_collate'),
        'SASRec': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'GRU4Rec': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'NextItNet': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'FSASRec': ('SEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'MF': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),        
        'DIN': (('SampleTwoTowerTrainDataset','sampletower_train_collate'), 'CandiEvalDataset', 'candi_eval_collate'),
        'DSSM': (('SampleTwoTowerTrainDataset','sampletower_train_collate'), 'SeqEvalDataset', 'seq_eval_collate'),
        'FM': (('SampleOneTowerTrainDataset','sampletower_train_collate'), 'SeqEvalDataset', 'seq_eval_collate'),
        'LightGCN': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),  
        'LightSANs': ('TwoTowerTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'SRGNN': (('GraphTrainDataset', 'graph_train_collate'), 'GraphEvalDataset', 'graph_eval_collate'),
        'VISRANK': ('BaseDataset', 'VisRankEvalDataset', 'base_collate'),
        'VBPR': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),
        'CuratorNet': ('TwoTowerTrainDataset2', 'SeqEvalDataset', 'seq_eval_collate'),
        'DVBPR': ('MOPairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),
        'ACF': ('SampleACFTrainDataset', 'ACFEvalDataset', 'seq_eval_collate'),
        'MOMF': ('MOPairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),
        'MOSASRec': ('MOSEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'MODIN': ('MOTwoTowerTrainDataset', 'CandiEvalDataset', 'candi_eval_collate'),
        'MOFM': (('MOSampleOneTowerTrainDataset','mosampletower_train_collate'), 'SeqEvalDataset', 'seq_eval_collate'),    
        'MOGRU4Rec': ('MOSEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'MONextItNet': ('MOSEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'MOBERT4Rec': ('MOBERT4RecTrainDataset', 'SeqEvalDataset','seq_eval_collate'),
        'MODSSM': (('MOSampleTwoTowerTrainDataset','mosampletower_train_collate'), 'SeqEvalDataset', 'seq_eval_collate'),
        'CESASRec': ('CEMOSEQTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'MOSRGNN': (('MOGraphTrainDataset', 'mograph_train_collate'), 'GraphEvalDataset', 'graph_eval_collate'),
        'MOLightSANs': ('MOTwoTowerTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
    }
 

    model_name = config['model']
    dataload.build()
       
    dataset_module = importlib.import_module('REC.data.dataset')
    train_set_name, test_set_name, collate_fn_name = dataset_dict[model_name]
     
    if isinstance(train_set_name, tuple):
        train_set_class = getattr(dataset_module, train_set_name[0])
        train_collate_fn = getattr(dataset_module, train_set_name[1]) 
    else:
        train_set_class = getattr(dataset_module, train_set_name) 
        train_collate_fn = None        
    
    test_set_class = getattr(dataset_module, test_set_name)
    eval_collate_fn = getattr(dataset_module, collate_fn_name)
 

    train_data = train_set_class(config,dataload)
    valid_data = test_set_class(config, dataload, phase='valid')
    test_data = test_set_class(config, dataload, phase='test')
    
        
    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') 
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') 
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    valid_sampler = NonConsecutiveSequentialDistributedSampler(valid_data)
    test_sampler = NonConsecutiveSequentialDistributedSampler(test_data)

    num_workers = 10
    rank = torch.distributed.get_rank() 
    seed = torch.initial_seed()
    
    init_fn = partial( 
    worker_init_fn, num_workers=num_workers, rank=rank, 
    seed=seed
    )
    
    
    if train_collate_fn:
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=train_sampler, collate_fn = train_collate_fn , worker_init_fn=init_fn)
    else:
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=train_sampler, worker_init_fn=init_fn)
    valid_loader = DataLoader(valid_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=valid_sampler, collate_fn=eval_collate_fn)

    test_loader = DataLoader(test_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=test_sampler,collate_fn=eval_collate_fn)

    return train_loader, valid_loader, test_loader



def worker_init_fn(worker_id, num_workers, rank, seed): 
    # The seed of each worker equals to 
    # num_worker * rank + worker_id + user_seed 
    worker_seed = num_workers * rank + worker_id + seed 
    np.random.seed(worker_seed) 
    random.seed(worker_seed)



def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    worker_seed = initial_seed + worker_id + torch.distributed.get_rank()
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class NonConsecutiveSequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = len(self.dataset)
        self.num_samples = math.ceil(
            (self.total_size-self.rank)/self.num_replicas
        )
     
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class ConsecutiveSequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples



class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)



