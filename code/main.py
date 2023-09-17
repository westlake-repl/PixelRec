import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--config_file",nargs='+')
    
    args = parser.parse_args()
    device = args.device
    config_file = args.config_file 

    import random
    master_port = random.randint(1002,9999)

    nproc_per_node = len(device.split(','))
    
    if len(config_file) ==2:
        run_yaml = f"CUDA_VISIBLE_DEVICES='{device}'  python  -m torch.distributed.run --nproc_per_node {nproc_per_node} \
    --master_port {master_port} run.py --config_file {config_file[0]} {config_file[1]}"
    elif len(config_file) ==1:
        run_yaml = f"CUDA_VISIBLE_DEVICES='{device}'  python  -m torch.distributed.run --nproc_per_node {nproc_per_node} \
    --master_port {master_port} run.py --config_file {config_file[0]}"


    os.system(run_yaml)


 

