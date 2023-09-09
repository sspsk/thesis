#package imports
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
import json
import argparse
from smplx.lbs import vertices2joints
import cv2
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

loss = torch.zeros(len(eval_loader))

with torch.no_grad():
    for bn,batch in enumerate(eval_loader,start=1):

        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(DEVICE)

        res_pred,res_gt = model.eval_step(batch)

        #get angles
        body_pose_pred = res_pred.body_pose
        body_pose_gt= res_gt.body_pose

        if body_pose_gt.ndim < 3: #its axis-ange representation
            raise NotImplementedError("Calculating between rotmat and axis-angle is not supported yet.")
        
        #and calculate diff

        #6d rotation
        #body_pose_pred = body_pose_pred.reshape(body_pose_pred.shape[0],-1,3,3)[:,:,:,:2].flatten(2,3) #has shape of [B,N_joints,6]
        #body_pose_gt = body_pose_gt.reshape(body_pose_gt.shape[0],-1,3,3)[:,:,:,:2].flatten(2,3)#has shape of [B,N_joints,6]
        
        #pose_loss = F.mse_loss(body_pose_pred,body_pose_gt,reduction='none').sum(-1).mean(-1).sum()

        #axis-angle
        body_pose_pred = body_pose_pred.reshape(-1,3,3)
        body_pose_pred_aa = torch.empty(body_pose_pred.shape[0],3,device=body_pose_pred.device)

        body_pose_gt = body_pose_gt.reshape(-1,3,3)
        body_pose_gt_aa = torch.empty(body_pose_gt.shape[0],3,device=body_pose_gt.device)

        for i in range(body_pose_pred_aa.shape[0]):
            body_pose_pred_aa[i] = torch.from_numpy(cv2.Rodrigues(body_pose_pred[i].detach().cpu().numpy())[0]).to(body_pose_pred.device).squeeze(-1)
            body_pose_gt_aa[i] = torch.from_numpy(cv2.Rodrigues(body_pose_gt[i].detach().cpu().numpy())[0]).to(body_pose_gt.device).squeeze(-1)
        
        body_pose_pred_aa = body_pose_pred_aa.reshape(-1,23,3)
        body_pose_gt_aa = body_pose_gt_aa.reshape(-1,23,3)

        pose_loss = F.mse_loss(body_pose_pred_aa,body_pose_gt_aa,reduction='none').sum(-1).mean(-1).sum(-1)
            
        
        loss[bn-1] = pose_loss

        print("Batch: {0}/{1}".format(bn,len(eval_loader)),end='\r')
        

print()
print("Rotation error:",loss.sum()/len(eval_dataset)) 

results_path = os.path.join(cfg['metadata']['exp_dir'],cfg['metadata']['name'],'rotation_results.json')
with open(results_path,'w') as f:
    json.dump({'mpre':(loss.sum()/len(eval_dataset)).tolist()},f)

print("Results saved at:",results_path)