#package imports
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import os
import json
import argparse
from smplx.lbs import vertices2joints
import numpy as np


#local imports
from data.eval_dataset import Dataset_3DPW,Dataset_MPII
import config
from data.utils import rot6d_to_rotmat,reconstruction_error_per_part
from exp_tracking.tracking_utils import get_checkpoint_path, get_model_class,parse_config
import models
import constants
from models.smpl import get_smpl_model


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
parser.add_argument('--type',default='best',help='Opt to eval a best/check/last checkpoint.')
parser.add_argument('--eval_mpii',action='store_true',help='Eval on MPII(shape related experiments)')
args = parser.parse_args()

cfg = parse_config(args.config_file)

if args.eval_mpii:
    eval_dataset = Dataset_MPII()
    print("Eval Dataset: MPII")
else:
    eval_dataset = Dataset_3DPW()
    print("Eval Dataset: 3DPW")
    
eval_loader = DataLoader(eval_dataset,batch_size=cfg['training']['bs'],shuffle=False,num_workers=8)
print("Dataset len:",len(eval_dataset))

checkpoint_path = get_checkpoint_path = get_checkpoint_path(cfg,type=args.type)
model_class = get_model_class(cfg)

model = model_class(cfg=cfg)


if checkpoint_path is not None:
    print("Loading checkpoint from:",checkpoint_path)
    chkpt = torch.load(checkpoint_path)
    try:
        model.load_state_dict(chkpt['model_state_dict'])
    except:
        print("LOG:: Model loading failed. Trying with strict=False. If not expected check the model.")
        model.load_state_dict(chkpt['model_state_dict'],strict=False)
    epochs = chkpt['epochs']
    print("Done")

if epochs is not None:
    print("Epochs:",epochs)

model.smpl_male = get_smpl_model(gender='male')

if torch.cuda.is_available() and cfg['training']['cuda']:
    DEVICE='cuda'
else:
    DEVICE='cpu'
print("Device used:",DEVICE)
        

model = model.to(DEVICE)
model.eval()

j36m_regressor= torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).to(dtype=torch.float32)

parts_error = torch.zeros(14)

with torch.no_grad():
    for bn,batch in enumerate(eval_loader,start=1):

        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(DEVICE)

        res_pred,res_gt = model.eval_step(batch)

        pred_vertices = res_pred.vertices   
        pred_h36m_joints = vertices2joints(j36m_regressor.to(pred_vertices.device),pred_vertices)
        pred_joints = pred_h36m_joints[:,constants.H36M_TO_J14,:]

        gt_vertices = res_gt.vertices   
        gt_h36m_joints = vertices2joints(j36m_regressor.to(pred_vertices.device),gt_vertices)
        gt_joints = gt_h36m_joints[:,constants.H36M_TO_J14,:]

        loss =  reconstruction_error_per_part(pred_joints.cpu().numpy(),gt_joints.cpu().numpy())

        parts_error += torch.from_numpy(loss).sum(dim=0)
        print("Batch: {0}/{1}".format(bn,len(eval_loader)),end='\r')
        

print()
print("Reconstruction error:",parts_error/len(eval_dataset)) 

results_path = os.path.join(cfg['metadata']['exp_dir'],cfg['metadata']['name'],'per_joints_results.json')
with open(results_path,'w') as f:
    json.dump({'pampjpe':(parts_error/len(eval_dataset)).tolist()},f)

print("Results saved at:",results_path)