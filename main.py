from data.eft_dataset import BaseDataset
from custom_logging.logger import Logger
from custom_logging.logging_utils import get_commit,get_branch
from models.smpl import get_smpl_model

bd = BaseDataset()
l = Logger()
cmt = get_commit()
print("Commit:",cmt)
br = get_branch()
print("Branch:",br)

mysmpl = get_smpl_model()
output = mysmpl()
print("Joints shape:",output.joints.shape)
print("Vertices shape:",output.vertices.shape)

print("exiting...")
