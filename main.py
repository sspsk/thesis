from Data.base_dataset import BaseDataset
from Logging.logging import Logger
from Logging.logging_utils import get_commit,get_branch
from Models.smpl import get_smpl_model

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
