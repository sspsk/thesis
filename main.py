from exp_tracking.logger import Logger
from exp_tracking.tracking_utils import get_commit,get_branch, parse_config, is_repo_clean
from models.smpl import get_smpl_model

cmt = get_commit()
print("Commit:",cmt)
br = get_branch()
print("Branch:",br)

mysmpl = get_smpl_model()
output = mysmpl()
print("Joints shape:",output.joints.shape)
print("Vertices shape:",output.vertices.shape)

cfg = parse_config("exp_config.yml")
print(cfg)

print("Is repo clean?",is_repo_clean())

print("exiting...")
