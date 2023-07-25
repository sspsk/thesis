import config
from data.utils import load_mean_parameters,rot6d_to_rotmat,reconstruction_error
from models.smpl import get_smpl_model

import torch
from torch.optim import Adam
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import sys


class Regressor(nn.Module):

    def __init__(self,cfg={}):
        super().__init__()
        self.cfg=cfg
        self.num_preds = 3 + 24*6 + 10
        self.theta_order = cfg.get("order","psc")

        theta_mean = load_mean_parameters(config.SMPL_MEAN_PARAMS,rot6d=True,order=self.theta_order)#load the real theta_mean
        self.register_buffer('theta_mean',theta_mean)
        
        self.relu = nn.ReLU()
        self.shape_layers = nn.Sequential(nn.Linear(2048+10,512),self.relu,nn.Dropout(),nn.Linear(512,512),self.relu,nn.Dropout(),nn.Linear(512,10))
        self.layers = nn.Sequential(
                nn.Linear(2048+self.num_preds,1024),self.relu,nn.Dropout(),
                nn.Linear(1024,1024),self.relu,nn.Dropout(),
                nn.Linear(1024,self.num_preds-10)
                )
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.01)

        norelu = cfg.get('norelu',False)
        if norelu:
            self.layers[1] = torch.nn.Identity()
            self.layers[4] = torch.nn.Identity()
        
        print("MODEL REGRESSOR ARCHITECTURE")
        print(self.layers)
        print(self.shape_layers)

    def forward(self,x,shape=None):
        theta = self.theta_mean.repeat(x.shape[0],1)
        pose = theta[:,:-13]
        cam = theta[:,-3:]

        cam_res = []
        pose_res = []
        shape_res = []

        if shape is None:
            shape = theta[:,-13:-3]#parameters are initialized from the mean parameters
            for i in range(3):
                input = torch.cat([x,shape],dim=1)
                shape = shape + self.shape_layers(input)
                shape_res.append(shape)
        else:
            shape_res.append(shape)

        for i in range(3):
            input = torch.cat([x,pose,shape,cam],dim=1)
            residual = self.layers(input)
            pose = pose + residual[:,:-3]
            cam = cam + residual[:,-3:]

            pose_res.append(pose)
            cam_res.append(cam)
            
        return [cam_res,pose_res,shape_res]


class HMR2(nn.Module):

    def __init__(self,cfg={}):
        super().__init__()
        self.regressor = Regressor(cfg=cfg)
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.encoder.fc = nn.Identity()
        self.cfg=cfg
        self.smpl = get_smpl_model()

        
    def forward(self,x,return_feature=False,shape=None):
        enc_feat = self.encoder(x)
        
        if return_feature:
            return self.regressor(enc_feat,shape=shape) + [enc_feat]
        else:
            return self.regressor(enc_feat,shape=shape)

    def train_step(self,batch,criterion):
        #assume that batch and model are on the same device
        loss_dict = {}
        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']

        if self.cfg['model'].get('gtshape',False):
            _,pose_pred,shape_pred = self(img,shape=shape_gt)
        else:
            _,pose_pred,shape_pred = self(img)

        mat_pose = []
        mat_pose.append(rot6d_to_rotmat(pose_pred[-1].reshape(-1,6)).reshape(-1,24,3,3))

        pose_loss = criterion[0](mat_pose[-1],pose_gt) 
        loss_dict['pose_loss'] = pose_loss

        shape_loss = criterion[0](shape_pred[-1],shape_gt)

        res_pred = self.smpl(global_orient=mat_pose[-1].flatten(2,3)[:,:1,:],
                             body_pose=mat_pose[-1].flatten(2,3)[:,1:,:],
                             betas=shape_pred[-1],
                             pose2rot=False)

        res_gt = self.smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                             body_pose=pose_gt.flatten(2,3)[:,1:,:],
                             betas=shape_gt,
                             pose2rot=False)
        
        joints_loss = criterion[1](res_pred.joints[:,:24,:],res_gt.joints[:,:24,:]).sum([1,2]).mean()
        loss_dict['joints_loss'] = joints_loss

        if self.cfg['model'].get('gtshape',False):
            loss = pose_loss + joints_loss
        else:
            if  self.cfg['model'].get('with_shape_loss',True):
                loss = pose_loss + shape_loss + joints_loss
                loss_dict['shape_loss'] = shape_loss
            else:
                loss = pose_loss + joints_loss


        return loss, loss_dict

        
    def validation_step(self,batch,criterion):
        #assume that batch and model are on the same device
        loss_dict = {}

        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']

        if self.cfg['model'].get('gtshape',False):
            _,pose_pred,shape_pred = self(img,shape=shape_gt)
        else:
            _,pose_pred,shape_pred = self(img)

        mat_pose = []
        mat_pose.append(rot6d_to_rotmat(pose_pred[-1].reshape(-1,6)).reshape(-1,24,3,3))

        res_pred = self.smpl(global_orient=mat_pose[-1].flatten(2,3)[:,:1,:],
                             body_pose=mat_pose[-1].flatten(2,3)[:,1:,:],
                             betas=shape_pred[-1],
                             pose2rot=False)

        res_gt = self.smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                             body_pose=pose_gt.flatten(2,3)[:,1:,:],
                             betas=shape_gt,
                             pose2rot=False)
        
        loss = criterion[1](res_pred.joints[:,:24,:],res_gt.joints[:,:24,:]).sum([1,2]).mean()
        loss_dict['joints_loss'] = loss


        return loss, loss_dict
    
    def eval_step(self,batch):
        #return the sum of error for the batch

        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']


        _,pose,shape = self(img)

        pose_pred = rot6d_to_rotmat(pose[-1].reshape(-1,6)).reshape(-1,24,3,3).flatten(2,3)
        shape_pred = shape[-1]


        res_pred = self.smpl(global_orient=pose_pred[:,:1,:],
                             body_pose=pose_pred[:,1:,:],
                             betas=shape_pred,
                             pose2rot=False)

        res_gt = self.smpl(global_orient=pose_gt[:,:3],body_pose=pose_gt[:,3:],betas=shape_gt,pose2rot=True)
        
        return reconstruction_error(res_pred.joints[:,:24].cpu().numpy(),res_gt.joints[:,:24].cpu().numpy(),reduction='sum')





    def get_optimizer(self):
        return Adam(params=self.parameters(),lr=self.cfg['training']['lr'])
    
    def get_criterion(self):
        #can be a single criterion or a list of them
        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss(reduction='none')
        return [criterion1,criterion2]