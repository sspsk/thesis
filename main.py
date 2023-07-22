from exp_tracking.logger import Logger
from exp_tracking.tracking_utils import get_commit,get_branch, parse_config, is_repo_clean, begin_experiment
from models.smpl import get_smpl_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
args = parser.parse_args()

cfg,curr_exp_dir =  begin_experiment(args.config_file,force=args.force)

print("Experiment done. Check logs at:",curr_exp_dir)


