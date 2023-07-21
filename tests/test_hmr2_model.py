from models.HMR2 import HMR2
import torch

def test_HMR2_model():
    config_dict = {}
    model = HMR2(config_dict=config_dict)

    dummy_img = torch.randn(1,3,224,224)
    dummy_pose = torch.randn(1,24,3,3)
    dummy_shape = torch.randn(1,10)

    batch = dict(img=dummy_img,pose=dummy_pose,shape=dummy_shape)
    criterions = model.get_criterion()

    model.train()
    loss = model.train_step(batch,criterions)
    assert len(loss.shape) == 0
    print("Train loss:",loss)

    model.eval()
    with torch.no_grad():
        val_loss = model.validation_step(batch,criterions)
    assert len(val_loss.shape) == 0
    print("Val loss:",val_loss)

    print("HMR2 model tests completed successfully.")