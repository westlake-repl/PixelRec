general_arguments = [
    'seed',
    'reproducibility',
    'state',
    'model',
    'data_path',
    'checkpoint_dir',
    'show_progress',
    'config_file',
    'log_wandb',
    'use_modality'
]

training_arguments = [
    'epochs', 'train_batch_size',
    'optim_args',
    'eval_step', 'stopping_step',
    'clip_grad_norm',
    'loss_decimal_place',
]

evaluation_arguments = [
    'eval_type',
    'repeatable',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place',
]

dataset_arguments = [
    'MAX_ITEM_LIST_LENGTH'
]





