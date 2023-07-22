import subprocess
import yaml

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
