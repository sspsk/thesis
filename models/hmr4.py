import config
import constants
from data.utils import load_mean_parameters,rot6d_to_rotmat,orth_proj,reconstruction_error
from models.backbone import CustomResNet
from models.smpl import get_smpl_model

import torch
from torch.optim import Adam
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from smplx.lbs import vertices2joints


class Regressor(nn.Module):

    def __init__(self,cfg={}):
        super().__init__()
        self.cfg=cfg
        self.num_preds = 3 + 24*6 + 10
        self.theta_order = cfg.get("order","psc")

        theta_mean = load_mean_parameters(config.SMPL_MEAN_PARAMS,rot6d=True,order=self.theta_order)#load the real theta_mean
        self.register_buffer('theta_mean',theta_mean)

        self.base_smpl = get_smpl_model()
        pose = theta_mean[:,:-13]
        pose = rot6d_to_rotmat(pose.reshape(-1,6)).reshape(-1,24,3,3).flatten(2,3)
        shape = theta_mean[:,-13:-3]


        with torch.no_grad():
            base_smpl = self.base_smpl(global_orient=pose[:,:1,:],
                                     body_pose=pose[:,1:,:],
                                     betas=shape,
                                     pose2rot=False)
            base_smpl_joints = base_smpl.joints[:,1:24,:] #has shape of [b,23,3], take all except the pelvis

            parents = self.base_smpl.parents[1:].unsqueeze(-1).unsqueeze(0)# has shape of [1,23,1]
            diff = base_smpl_joints - torch.take_along_dim(base_smpl.joints,parents,dim=1)

            joints_feat = torch.linalg.norm(diff,dim=-1)
            self.register_buffer('mean_joints_feat',joints_feat)
            del self.base_smpl
        
        self.relu = nn.ReLU()
        self.joints_layers = nn.Sequential(nn.Linear(2048+23,512),nn.Dropout(),nn.Linear(512,512),nn.Dropout(),nn.Linear(512,23))
        self.layers = nn.Sequential(
                nn.Linear(2048+23+self.num_preds,1024),self.relu,nn.Dropout(),
                nn.Linear(1024,1024),self.relu,nn.Dropout(),
                nn.Linear(1024,self.num_preds)
                )
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.01)

        norelu = cfg['model'].get('norelu',False)
        if norelu:
            self.layers[1] = torch.nn.Identity()
            self.layers[4] = torch.nn.Identity()
        
        print("MODEL REGRESSOR ARCHITECTURE")
        print(self.layers)

    def forward(self,x):
        theta = self.theta_mean.repeat(x.shape[0],1)
        joints_feat = self.mean_joints_feat.repeat(x.shape[0],1)

        for i in range(3):
            input = torch.cat([x,joints_feat],dim=1)
            joints_feat = joints_feat + self.joints_layers(input)

        cam_res = []
        pose_res = []
        shape_res = []

        for i in range(3):
            input = torch.cat([x,joints_feat,theta],dim=1)
            theta = theta + self.layers(input)
            
            if self.theta_order == "psc":
                cam_res.append(theta[:,-3:])
                pose_res.append(theta[:,:-13])
                shape_res.append(theta[:,-13:-3])
            elif self.theta_order == "cps":
                cam_res.append(theta[:,:3])
                pose_res.append(theta[:,3:-10])
                shape_res.append(theta[:,-10:])
                
        return [cam_res,pose_res,shape_res,joints_feat]


class HMR4(nn.Module):

    def __init__(self,cfg={}):
        super().__init__()
        self.regressor = Regressor(cfg=cfg)
        if cfg['model'].get('custom_backbone',False):
            print("LOG:: Using custom BackBone")
            self.encoder = CustomResNet()
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder.load_state_dict(pretrained_model.state_dict(),strict=False)
        else:
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Identity()
        self.cfg=cfg
        if cfg['model'].get('use_extra_smpl',False):
            self.smpl = get_smpl_model(use_feet_keypoints=True,use_hands=True,extra=True)
        else:
            self.smpl = get_smpl_model()
        self.base_smpl = get_smpl_model()

        
    def forward(self,x,return_feature=False,joints_feat=None):
        enc_feat = self.encoder(x)
        
        if return_feature:
            return self.regressor(enc_feat) + [enc_feat]
        else:
            return self.regressor(enc_feat)

    def train_step(self,batch,criterion):
        #assume that batch and model are on the same device
        loss_dict = {}
        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']
        gt_kp = batch['keypoints2d']
        vis = batch['visibility2d']

        #on the fly get the joints distance features
        with torch.no_grad():
            base_smpl = self.base_smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                                 body_pose=pose_gt.flatten(2,3)[:,1:,:],
                                 betas=shape_gt,
                                 pose2rot=False)
            base_smpl_joints = base_smpl.joints[:,1:24,:] #has shape of [b,23,3], take all except the pelvis

            parents = self.base_smpl.parents[1:].unsqueeze(-1).unsqueeze(0)# has shape of [1,23,1]
            diff = base_smpl_joints - torch.take_along_dim(base_smpl.joints,parents,dim=1)

            gt_joints_feat = torch.linalg.norm(diff,dim=-1)



        cam_pred,pose_pred,shape_pred,joints_feat_pred = self(img)

        mat_pose = []
        mat_pose.append(rot6d_to_rotmat(pose_pred[-1].reshape(-1,6)).reshape(-1,24,3,3))

        pose_loss = criterion[0](mat_pose[-1],pose_gt) 
        loss_dict['pose_loss'] = pose_loss


        res_pred = self.smpl(global_orient=mat_pose[-1].flatten(2,3)[:,:1,:],
                             body_pose=mat_pose[-1].flatten(2,3)[:,1:,:],
                             betas=shape_pred[-1],
                             pose2rot=False)

        res_gt = self.smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                             body_pose=pose_gt.flatten(2,3)[:,1:,:],
                             betas=shape_gt,
                             pose2rot=False)
        
        joints_loss = criterion[1](res_pred.joints,res_gt.joints).sum([1,2]).mean()#3d keypoints loss on all 49 joints of custom smpl
        loss_dict['joints_loss'] = joints_loss
        loss = pose_loss + joints_loss

        if self.cfg['model'].get('with_shape_loss',True):
            shape_loss = criterion[0](shape_pred[-1],shape_gt)
            loss += shape_loss
            loss_dict['shape_loss'] = shape_loss
        
        if self.cfg['model'].get('with_reprojection_loss',True):
            pred_kp = orth_proj(res_pred.joints,cam_pred[-1])
            #normalizing gt_kp to [-1,1]
            gt_kp = gt_kp/(constants.IMG_RES/2) - 1.0
            reprojection_loss = (criterion[1](pred_kp,gt_kp) * vis.unsqueeze(-1)).sum([1,2]).mean()
            loss += reprojection_loss
            loss_dict['reprojection_loss'] = reprojection_loss
        
        if self.cfg['model'].get('with_jointsfeat_loss',True):
            jointsfeat_loss = criterion[0](joints_feat_pred,gt_joints_feat)
            loss += jointsfeat_loss
            loss_dict['jointsfeat_loss'] = jointsfeat_loss

        return loss, loss_dict

        
    def validation_step(self,batch,criterion):
        #assume that batch and model are on the same device

        if not hasattr(self,'j36m_regressor'):
            self.j36m_regressor= torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).to(dtype=torch.float32)
        loss_dict = {}

        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']

        with torch.no_grad():
            base_smpl = self.base_smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                                 body_pose=pose_gt.flatten(2,3)[:,1:,:],
                                 betas=shape_gt,
                                 pose2rot=False)
            base_smpl_joints = base_smpl.joints[:,1:24,:] #has shape of [b,23,3], take all except the pelvis

            parents = self.base_smpl.parents[1:].unsqueeze(-1).unsqueeze(0)# has shape of [1,23,1]
            diff = base_smpl_joints - torch.take_along_dim(base_smpl.joints,parents,dim=1)

            joints_feat = torch.linalg.norm(diff,dim=-1)

        _,pose_pred,shape_pred,_ = self(img)

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
        

        pred_vertices = res_pred.vertices   
        pred_h36m_joints = vertices2joints(self.j36m_regressor.to(pred_vertices.device),pred_vertices)
        pred_joints = pred_h36m_joints[:,constants.H36M_TO_J14,:]

        gt_vertices = res_gt.vertices   
        gt_h36m_joints = vertices2joints(self.j36m_regressor.to(pred_vertices.device),gt_vertices)
        gt_joints = gt_h36m_joints[:,constants.H36M_TO_J14,:]

        loss =  torch.tensor(reconstruction_error(pred_joints.cpu().numpy(),gt_joints.cpu().numpy(),reduction='mean'))
        loss_dict['joints_loss'] = loss

        return loss, loss_dict
    
    def eval_step(self,batch):
        #return the gt & predicted smpl joints

        img = batch['img']
        pose_gt = batch['pose']
        shape_gt = batch['shape']

        if pose_gt.ndim < 4:#some datasets(3dpw) return axis-angle
            res_gt = self.smpl_male(global_orient=pose_gt[:,:3],body_pose=pose_gt[:,3:],betas=shape_gt,pose2rot=True)
        else:#others return matrices
            res_gt = self.base_smpl(global_orient=pose_gt.flatten(2,3)[:,:1,:],
                                 body_pose=pose_gt.flatten(2,3)[:,1:,:],
                                 betas=shape_gt,
                                 pose2rot=False)

        with torch.no_grad():

            base_smpl_joints = res_gt.joints[:,1:24,:] #has shape of [b,23,3], take all except the pelvis

            parents = self.base_smpl.parents[1:].unsqueeze(-1).unsqueeze(0)# has shape of [1,23,1]
            diff = base_smpl_joints - torch.take_along_dim(res_gt.joints,parents,dim=1)

            joints_feat = torch.linalg.norm(diff,dim=-1)


        cam,pose,shape,_= self(img)

        pose_pred = rot6d_to_rotmat(pose[-1].reshape(-1,6)).reshape(-1,24,3,3).flatten(2,3)
        shape_pred = shape[-1]


        res_pred = self.smpl(global_orient=pose_pred[:,:1,:],
                             body_pose=pose_pred[:,1:,:],
                             betas=shape_pred,
                             pose2rot=False)


        return res_pred,res_gt


    def get_optimizer(self):
        return Adam(params=self.parameters(),lr=self.cfg['training']['lr'])
    
    #def get_criterion(self):
    #    #can be a single criterion or a list of them
    #    criterion1 = nn.MSELoss()
    #    criterion2 = nn.MSELoss(reduction='none')
    #    return [criterion1,criterion2]
    def get_criterion(self):
        #can be a single criterion or a list of them
        if self.cfg['model'].get('l1_loss',False):
            criterion1 = nn.L1Loss()
            criterion2 = nn.L1Loss(reduction='none')
        else:
            criterion1 = nn.MSELoss()
            criterion2 = nn.MSELoss(reduction='none')
        return [criterion1,criterion2]