from models.hmr2 import HMR2
from exp_tracking.tracking_utils import parse_config
import torch

def test_HMR2_model():
    cfg = parse_config('exp_config.yml')
    model = HMR2(cfg=cfg)

    dummy_img = torch.randn(1,3,224,224)
    dummy_pose = torch.randn(1,24,3,3)
    dummy_shape = torch.randn(1,10)

    batch = dict(img=dummy_img,pose=dummy_pose,shape=dummy_shape)
    criterions = model.get_criterion()
    optimizer = model.get_optimizer()

    model.train()
    loss,loss_dict = model.train_step(batch,criterions)
    assert len(loss.shape) == 0
    print("Train loss:",loss)

    model.eval()
    with torch.no_grad():
        val_loss,loss_dict = model.validation_step(batch,criterions)
    assert len(val_loss.shape) == 0
    print("Val loss:",val_loss)

    batch['pose'] = torch.randn(1,72)
    with torch.no_grad():
        eval_loss = model.eval_step(batch)
    assert len(eval_loss.shape) == 1
    print("Eval Loss shape:",eval_loss.shape)


    print("HMR2 model tests completed successfully.")