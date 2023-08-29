import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

import config
import constants
from data.utils import load_mean_parameters,rot6d_to_rotmat,orth_proj
from models.flows import CondFlowBlock
from models.backbone import CustomResNet
from models.smpl import get_smpl_model

class ProHMR(nn.Module):
    def __init__(self,cfg={}):
        super().__init__()
        self.feat_dim = 6*24#rot6d for 24 SMPL joints
        self.cond_dim=2048#the dim or last resnet layer
        self.theta_order=config.get('theta_order','psc')

        theta_mean = load_mean_parameters(config.SMPL_MEAN_PARAMS,rot6d=True,order=self.theta_order)#load the real theta_mean

        cam_mean = theta_mean[:,-3]#last three params
        shape_mean = theta_mean[:,-13:-3]#last ten params

        self.register_buffer('theta_mean',theta_mean)
        self.register_buffer('cam_mean',cam_mean)
        self.register_buffer('shape_mean',shape_mean)

        if cfg['model'].get('custom_backbone',False):
            print("LOG:: Using custom BackBone")
            self.encoder = CustomResNet()
            pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder.load_state_dict(pretrained_model.state_dict(),strict=False)
        else:
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Identity()

        if cfg['model'].get('use_extra_smpl',False):
            self.smpl = get_smpl_model(use_feet_keypoints=True,use_hands=True,extra=True)
        else:
            self.smpl = get_smpl_model()

        self.cfg=cfg

        self.flows = nn.ModuleList([CondFlowBlock(feat_dim=self.feat_dim,cond_dim=self.cond_dim),CondFlowBlock(feat_dim=self.feat_dim,cond_dim=self.cond_dim),
                                    CondFlowBlock(feat_dim=self.feat_dim,cond_dim=self.cond_dim),CondFlowBlock(feat_dim=self.feat_dim,cond_dim=self.cond_dim)])

        self.head = nn.Sequential(nn.Linear(2048,1024),
                                  nn.ReLU(),
                                  nn.Linear(1024,10+3))

        nn.init.xavier_uniform_(self.head[2].weight, gain=0.02)

    def forward(self,img,pose,n_samples=1):
        if torch.isnan(img).any():
            print("Found nan in img")
            import pdb;pdb.set_trace()
            

        img_feats = self.encoder(img)
        if torch.isnan(img_feats).any():
            print("Found nan in img_feats")
            import pdb;pdb.set_trace()
        B = img.shape[0]

        z = pose
        log_det = 0
        for flow in self.flows:
            z,ld = flow(z,img_feats)
            log_det += ld

        offset_shape_cam = self.head(img_feats)
        offset_cam = offset_shape_cam[:,-3:]
        offset_shape = offset_shape_cam[:,:10]

        cam = self.cam_mean + offset_cam
        shape = self.shape_mean + offset_shape

        mode_pose = torch.zeros(B,144,device=img_feats.device)
        for f in self.flows[::-1]:
            mode_pose,_ = f(mode_pose,img_feats,reverse=True)

        samples=None
        if n_samples > 1:
            samples = torch.randn(B*(n_samples-1),144,device=img_feats.device)        
            multi_img_feats = img_feats.unsqueeze(1).repeat(1,n_samples-1,1).reshape(B*(n_samples-1),-1)#should be of shape (B*(n_samples-1),144)
            for f in self.flows[::-1]:
                samples,_ = f(samples,multi_img_feats,reverse=True)


        return z,log_det,cam,shape,mode_pose,samples

    def sample(self,img):
        img_feats = self.encoder(img)
        B = img.shape[0]

        z = torch.randn(B,144,device=img_feats.device)

        return self.inference(img,z)

    def get_criterion(self):
        #can be a single criterion or a list of them
        criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss(reduction='none')
        return [criterion1,criterion2]

    def get_optimizer(self):
        return Adam(params=( p for p in self.parameters() if p.requires_grad),lr=self.cfg['training']['lr'])

    def inference(self,img,z=None):
        img_feats = self.encoder(img)
        B = img.shape[0]

        if z is None:
            z = torch.zeros(B,144,device=img_feats.device)
        pose = z
        for f in self.flows[::-1]:
            pose,_ = f(pose,img_feats,reverse=True)

        offset_shape_cam = self.head(img_feats)
        offset_cam = offset_shape_cam[:,-3:]
        offset_shape = offset_shape_cam[:,:10]

        cam = self.cam_mean + offset_cam
        shape = self.shape_mean + offset_shape

        return pose,shape,cam

    def prob_criterion(self,z,ld):
        lp = -0.5*(torch.log(2*torch.tensor(np.pi)) + z**2)
        lp = lp.sum(dim=1)
        return lp.mean(),(lp + ld).mean()


    def train_step(self,batch,criterion):
        #perform inference and calculate loss

        img = batch['img'] 
        pose_gt = batch['pose']
        shape_gt = batch['shape']
        gt_kp = batch['keypoints2d']
        vis = batch['visibility2d']

        loss_dict = {}

        B = img.shape[0]

        #convert pose to 6d from rotmat
        rot6d_gt = pose_gt[:,:,:,:2].flatten(2,3)

        z,log_det,cam,shape,mode_pose,samples = self(img,pose_gt,n_samples=self.cfg.get('num_samples',3))

        _,log_prob = self.prob_criterion(z,log_det) 
        loss = -log_prob

        #samples has shape [b*n_samples-1,144]
        #mode_pose has shape [b,144]

        if samples is not None:
            n_samples = samples.shape[0]//B + 1
            shape = shape.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,10)
            mode_pose = mode_pose.unsqueeze(1)
            samples = samples.reshape(B,-1,self.feat_dim)
            poses = torch.cat([mode_pose,samples],dim=1).reshape(-1,self.feat_dim) #has shape [B*num_samples,144]
            pose_mat = rot6d_to_rotmat(poses.reshape(-1,6)).reshape(-1,24,3,3)
        else:
            pose_mat = rot6d_to_rotmat(mode_pose.reshape(-1,6)).reshape(-1,24,3,3)
            
        pose_loss = criterion[1](pose_mat,pose_gt).sum([1,2,3]).reshape(B,-1)#has shape of [B,n_samples]


        pose_pred = pose_mat.flatten(2,3)#has shape of [B*n_samples,24,9]

        if samples is not None:
            n_samples = samples.shape[0]//B + 1
            pose_gt = pose_gt.unsqueeze(1).repeat(1,n_samples,24,3,3).reshape(-1,24,3,3).flatten(2,3)
            shape_gt = shape_gt.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,10)
        else:
            pose_gt = pose_gt.flatten(2,3)

        res_pred = self.smpl(global_orient=pose_pred[:,:1,:],body_pose=pose_pred[:,1:,:],betas=shape,pose2rot=False)
        res_gt = self.smpl(global_orient=pose_gt[:,:1,:],body_pose=pose_gt[:,1:,:],betas=shape_gt,pose2rot=False)

        joints_loss = criterion[1](res_pred.joints,res_gt.joints).sum([1,2]).reshape(B,-1)#has shape of [B,n_samples]

        pose_loss_mode = pose_loss[:,0].mean()
        pose_loss_exp = pose_loss[:,1:].mean()

        loss_dict['pose_mode'] = pose_loss_mode
        loss_dict['pose_exp'] = pose_loss_exp
        
        joints_loss_mode = joints_loss[:,0].mean()
        joints_loss_exp = joints_loss[:,1:].mean()

        loss_dict['joints_loss_mode'] = joints_loss_mode
        loss_dict['joints_loss_exp'] = joints_loss_exp

        loss += pose_loss_mode + joints_loss_mode 

        if self.cfg['model'].get('with_shape_loss',True):
            shape_loss = criterion[0](shape,shape_gt)
            loss += shape_loss
            loss_dict['shape_loss'] = shape_loss
        
        if self.cfg['model'].get('with_reprojection_loss',True):
            pred_kp = orth_proj(res_pred.joints,cam) #has shape of [B*n_samples,49,2]
            #normalizing gt_kp to [-1,1]
            gt_kp = gt_kp/(constants.IMG_RES/2) - 1.0 #has shape of [B,49,2]

            if samples is not None:
                n_samples = samples.shape[0]//B + 1
                gt_kp = gt_kp.unsqueeze(1).repeat(1,n_samples,1,1).reshape(B*n_samples,-1.2)
                vis = vis.unsqueeze(1).repeat(1,n_samples,1).reshape(-1,2)

            reprojection_loss = (criterion[1](pred_kp,gt_kp) * vis.unsqueeze(-1)).sum([1,2]).reshape(B,-1) #has shape of [b,n_samples]

            reprojection_loss_mode = reprojection_loss[:,0].mean()
            reprojection_loss_exp = reprojection_loss[:,1:].mean()

            loss += reprojection_loss_mode + reprojection_loss_exp

            loss_dict['reprojection_loss_mode'] = reprojection_loss_mode
            loss_dict['reprojection_loss_exp'] = reprojection_loss_exp

        return loss, loss_dict





