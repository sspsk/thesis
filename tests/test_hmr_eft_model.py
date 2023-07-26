import torch

from models.hmr_eft import hmr_eft,hmr_eft_new
import config

def test_hmr_eft_model():
    model1,resnet1 = hmr_eft(config.SMPL_MEAN_PARAMS,pretrained=True,return_resnet=True)
    print("Model 1 initialized successfully.")

    model2,resnet2 = hmr_eft_new(config.SMPL_MEAN_PARAMS,pretrained=True,return_resnet=True)
    print("Model 2 initialized successfully.")

    params1 = resnet1.parameters() 
    params2 = resnet2.parameters() 

    for p1,p2 in zip(params1,params2):
        assert (p1==p2).all()
    
    print("Models checked and have the same weights.")

    dummy_img = torch.randn(1,3,224,224)
    pose,shape,cam = model2(dummy_img)

    assert len(pose.shape) == 4
    assert len(shape.shape) == 2
    assert len(cam.shape) == 2

    print("Dummy input test passed successfully.")

