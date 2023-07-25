import subprocess
import yaml
import os
import sys

def get_commit():
    return subprocess.check_output("git log -t".split(" ")).decode("utf-8").split("\n")[0].strip().split(" ")[1]

def get_branch():
    return subprocess.check_output("git branch".split(" ")).decode("utf-8").split("\n")[0].strip().split(" ")[1]

def is_repo_clean():
    return len(subprocess.check_output("git status --short".split(" ")).decode("utf-8")) == 0

def parse_config(config_filename):
    with open(config_filename,"r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def begin_experiment(config_filename,force=False):
    #check for changes, get config, create experiment folder, store generated_config

    uncommited_changes = False
    if not is_repo_clean():
        if force:
            print("Found uncommited changes. Proceed with caution.")
            uncommited_changes = True
        else:
            print("Found uncommited changes. Commit changes and try again.")
            exit()
    
    cfg = parse_config(config_filename)

    curr_exp_dir = os.path.join(cfg['metadata']['exp_dir'],cfg['metadata']['name'])
    try:
        os.mkdir(curr_exp_dir)
    except:
        print("This experiment maybe already exists. Check again.")
        exit()

    cfg['tracking'] = dict(hash=get_commit(),branch=get_branch(),uncommited_changes_at_the_time=uncommited_changes)
    
    with open(os.path.join(curr_exp_dir,"generated_config.yml"),"w") as f:
        yaml.dump(cfg,f)
    
    return cfg, curr_exp_dir

def get_model_class(cfg):
    class_str = cfg['model']['type'] 
    try:
        model_class = getattr(sys.modules['models'],class_str)
    except:
        print("The specified class: {0} does not exist. Check again.".format(class_str))
        exit()
    return model_class

def get_checkpoint_path(cfg,type='best'):
    if type == 'best':
        file_end = 'best.pt'
    elif type == 'check':
        file_end = 'check.pt'
    elif type == 'last':
        file_end == 'last.pt'
    else:
        print("Unkown type. Check again.")
        exit()
    
    metadata = cfg['metadata']
    chkpt_path = os.path.join(metadata['exp_dir'],metadata['name'],file_end)
    if os.path.exists(chkpt_path):
        return chkpt_path
    else:
        return None