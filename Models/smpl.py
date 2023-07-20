import smplx
import config


def get_smpl_model():
    return smplx.SMPL(model_path=config.SMPLX_PKG_MODELS+"smpl/",gender='neutral',use_hands=False,use_feet_keypoints=False) 