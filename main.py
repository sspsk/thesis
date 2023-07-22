from exp_tracking.tracking_utils import begin_experiment
from data.eft_dataset import EFTDataset
import config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
args = parser.parse_args()

cfg,curr_exp_dir =  begin_experiment(args.config_file,force=args.force)

datasets = cfg['data']['datasets']

dataset = EFTDataset(datasets=datasets)
print(len(dataset))
print("Dataset created successfully.")