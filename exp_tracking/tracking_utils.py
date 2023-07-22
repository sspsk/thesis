import subprocess
import yaml
import os

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

    if not is_repo_clean() and not force:
        print("Found uncommited changes. Commit changes and try again.")
        exit()
    
    cfg = parse_config(config_filename)

    curr_exp_dir = os.path.join(cfg['metadata']['exp_dir'],cfg['metadata']['name'])
    os.mkdir(curr_exp_dir)

    cfg['tracking'] = dict(hash=get_commit(),branch=get_branch())
    
    with open(os.path.join(curr_exp_dir,"generated_config.yml"),"w") as f:
        yaml.dump(cfg,f)
    
    return cfg, curr_exp_dir

    
