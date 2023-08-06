import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import random

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True #so far doesnt affect speed
#torch.backends.cudnn.benchmark = False

random.seed(42)
np.random.seed(42)

from exp_tracking.tracking_utils import begin_experiment,get_model_class,log_print
from data.eft_dataset import EFTDataset
import models
import config



parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
args = parser.parse_args()

cfg,curr_exp_dir =  begin_experiment(args.config_file,force=args.force)

data_cfg = cfg['data']
datasets = data_cfg['datasets']

train_dataset = EFTDataset(datasets=datasets,is_train=True,cfg=data_cfg)
train_loader = DataLoader(train_dataset,batch_size=cfg['training']['bs'],shuffle=True,pin_memory=True,num_workers=8)
log_print("Train Dataset length:",len(train_dataset),cfg=cfg)

val_dataset = EFTDataset(datasets=['coco14_val'],is_train=False,cfg=data_cfg)
val_loader = DataLoader(val_dataset,batch_size=cfg['training']['bs'],shuffle=False,pin_memory=True,num_workers=8)
log_print("Val Dataset length:",len(val_dataset),cfg=cfg)

if torch.cuda.is_available() and cfg['training']['cuda']:
    DEVICE='cuda'
else:
    DEVICE='cpu'
log_print("Device used:",DEVICE,cfg=cfg)

model_class = get_model_class(cfg=cfg)
log_print("Model class used:",model_class,cfg=cfg)

model = model_class(cfg=cfg)
model = model.to(DEVICE)

optimizer = model.get_optimizer()
criterion = model.get_criterion()

train_loss = float('inf')
val_loss = float('inf')
best_val_loss = float('inf')

epochs_done=0

checkpoint = cfg['training'].get("checkpoint",None)
if checkpoint is not None:

    #load model & optimizer
    log_print("Loading checkpoint from:",checkpoint,cfg=cfg)
    chkpt = torch.load(checkpoint)

    model.load_state_dict(chkpt['model_state_dict'])
    try:
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    except:
        log_print("Unable to load optimizer state. If you provided one, check again.",cfg=cfg)

    try:
        torch.set_rng_state(chkpt['torch_rng_state'])
        torch.cuda.set_rng_state(chkpt['torch_cuda_rng_state'])
        random.setstate(chkpt['python_rng_state'])
        np.random.set_state(chkpt['np_rng_state'])
    except:
        log_print("Unable to load rng states. If you provided states, check again.",cfg=cfg)
        
    epochs_done = chkpt['epochs']

    train_loss = chkpt['train_loss']
    val_loss = chkpt['val_loss']
    best_val_loss = chkpt['best_val_loss']

    log_print("Epochs done:",epochs_done,cfg=cfg)
    log_print("Previous Train Loss:",train_loss,cfg=cfg)
    log_print("Previous Val Loss:",val_loss,cfg=cfg)
    log_print("Previous Best Val Loss:",best_val_loss,cfg=cfg)

epochs = cfg['training']['epochs']

for ep in range(epochs_done+1,epochs_done + 1+ epochs):

    model.train()
    train_running_loss = 0.0
    for batch_num,batch in enumerate(train_loader,start=1):

        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(DEVICE)

        optimizer.zero_grad()

        loss, loss_dict = model.train_step(batch,criterion)
        loss.backward()

        optimizer.step()

        train_running_loss += loss.detach().cpu()
        if batch_num % 25 == 0:
            log_print("Epoch: {0}, {1}/{2}, Loss: {3}".format(ep,batch_num,len(train_loader),train_running_loss/batch_num),end=" ",cfg=cfg)
            for key in loss_dict:
                log_print("{0}: {1}".format(key,loss_dict[key]),end=" ",cfg=cfg)
            log_print(cfg=cfg)

    train_loss = train_running_loss/batch_num
    log_print("Epoch {0}: Train Loss: {1}".format(ep,train_loss),cfg=cfg)

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch_num,batch in enumerate(val_loader,start=1):

            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(DEVICE)

            loss,loss_dict = model.validation_step(batch,criterion)

            val_running_loss += loss.detach().cpu()

            log_print("Val {0}/{1}, Loss: {2}".format(batch_num,len(val_loader),val_running_loss/batch_num),end=" ",cfg=cfg)
            for key in loss_dict:
                log_print("{0}: {1}".format(key,loss_dict[key]),end=" ",cfg=cfg)
            log_print(end="\r",cfg=cfg)
    
    val_loss = val_running_loss/batch_num 
    log_print(cfg=cfg)
    log_print("Epoch {0}: Val Loss: {1}".format(ep,val_loss),cfg=cfg)

    if best_val_loss > val_loss or ep%10 == 0:

        is_best = False
        if best_val_loss > val_loss: 
            best_val_loss=val_loss
            is_best = True
        
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state(),
            "python_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state()
        }

        if is_best:
            best_filename = os.path.join(curr_exp_dir,"best.pt")
            log_print("Saving best model at:",best_filename,cfg=cfg)
            torch.save(checkpoint_dict,best_filename)
        if ep%10 == 0:
            check_filename = os.path.join(curr_exp_dir,"check.pt")
            log_print("Saving every-10-epochs checkpoint model at:",check_filename,cfg=cfg)
            torch.save(checkpoint_dict,check_filename)


#Saving the last checkpoint
checkpoint_dict = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs": epochs_done+epochs,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "best_val_loss": best_val_loss,
    "torch_rng_state": torch.get_rng_state(),
    "torch_cuda_rng_state": torch.cuda.get_rng_state(),
    "python_rng_state": random.getstate(),
    "np_rng_state": np.random.get_state()
}

last_filename = os.path.join(curr_exp_dir,"last.pt")
log_print("Saving last model at:",last_filename,cfg=cfg)
torch.save(checkpoint_dict,last_filename)

log_print("Training completed.",cfg=cfg)