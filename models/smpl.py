import smplx
import config
import os

def get_smpl_model(gender='neutral',use_hands=False,use_feet_keypoints=False):
    return smplx.SMPL(model_path=os.path.join(config.SMPLX_PKG_MODELS,"smpl/"),gender=gender,use_hands=use_hands,use_feet_keypoints=use_feet_keypoints) 