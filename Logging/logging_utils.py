import subprocess

def get_commit():
    return subprocess.check_output("git log -t".split(" ")).decode("utf-8").split("\n")[0].strip().split(" ")[1]

def get_branch():
    return subprocess.check_output("git branch".split(" ")).decode("utf-8").split("\n")[0].strip().split(" ")[1]

def parse_config(config_filename):
    pass