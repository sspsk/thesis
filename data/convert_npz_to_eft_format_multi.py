'''
Script to convert npz files to eft format where
data is a list of dicts.

EFT Keys:
-imageName
-bbox_scale
-bbox_center
-gt_keypoint_2d
-gt_keypoint_3d
-parm_pose
-parm_shape
-parm_cam

Npz Keys:
-imgname
-center
-scale
-part
-S
-openpose
-pose
-shape
-has_smpl
'''

from tqdm import tqdm
import json
import torch
import numpy as np
import sys
import os
from multiprocessing import Pool
sys.path.append("../")


import config
from data.utils import rodrigues_formula

def npz_to_mine(imgname):
    'correct imgname to match with my files'
    parts = imgname.split("/")
    img = parts[-1].split("_")[-1]
    return os.path.join(parts[0],parts[1],parts[3],img)


new_data = []

#def init_worker(data):
#    print("Initialized")
#    global shared_data
#    shared_data = data

def make_dict(idx):
    shared_data = np.load(config.MPIINF_ANNOT_FILE)
    i = idx
    newdict = {}

    imgname = npz_to_mine(shared_data['imgname'][i])
    center = shared_data['center'][i]
    scale = shared_data['scale'][i]
    part = shared_data['part'][i]
    op = shared_data['openpose'][i]
    S = shared_data['S'][i]
    pose = shared_data['pose'][i]
    shape = shared_data['shape'][i]
    has_smpl = shared_data['has_smpl'][i]

    newdict['imageName'] = imgname

    newdict['bbox_center'] = center.tolist()
    newdict['bbox_scale'] = float(scale)

    newdict['gt_keypoint_2d'] = np.concatenate([op,part],axis=0).tolist()
    newdict['gt_keypoint_3d'] = S.tolist()

    mat_pose = rodrigues_formula(torch.from_numpy(pose).to(torch.float32).unsqueeze(0)).squeeze(0)
    newdict['parm_pose'] = mat_pose.tolist()
    newdict['parm_shape'] = shape.tolist()
    newdict['parm_cam'] = [0.0,0.0,0.0] #just create a dummy camera
    newdict['has_smpl'] = float(has_smpl)

    return newdict

if __name__=='__main__':
    data = np.load(config.MPIINF_ANNOT_FILE)
    datalen = len(data['imgname'])


    with Pool(4) as p:
        new_data = list(tqdm(p.imap(make_dict,list(range(datalen))),total=datalen))

    outfile = '/gpu-data/skar/MPIINF/multi_mpiinf-eft.json'
    print("Saving json file at {0}....".format(outfile))
    with open(outfile,'w') as f:
        json.dump({'data':new_data},f)












