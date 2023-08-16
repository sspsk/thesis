import os
import torch
from smplx import SMPL as _SMPL
from smplx.lbs import vertices2joints
import numpy as np

import config
import constants

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        #J_regressor_h36m = np.load(config.JOINT_REGRESSOR_H36M)
        #self.register_buffer('J_regressor_h36m', torch.tensor(J_regressor_h36m, dtype=torch.float32))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        #h36m_joints = vertices2joints(self.J_regressor_h36m, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints,extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        smpl_output.joints = joints
        #output = ModelOutput(vertices=smpl_output.vertices,
        #                     global_orient=smpl_output.global_orient,
        #                     body_pose=smpl_output.body_pose,
        #                     joints=joints,
        #                     betas=smpl_output.betas,
        #                     full_pose=smpl_output.full_pose)
        return smpl_output


def get_smpl_model(gender='neutral',use_hands=False,use_feet_keypoints=False,extra=False):
    if extra:
        return SMPL(model_path=os.path.join(config.SMPLX_PKG_MODELS,"smpl/"),gender=gender,use_hands=use_hands,use_feet_keypoints=use_feet_keypoints) 
    else:
        return _SMPL(model_path=os.path.join(config.SMPLX_PKG_MODELS,"smpl/"),gender=gender,use_hands=use_hands,use_feet_keypoints=use_feet_keypoints) 