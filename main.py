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

from exp_tracking.tracking_utils import begin_experiment,get_model_class
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
print("Train Dataset length:",len(train_dataset))

val_dataset = EFTDataset(datasets=['coco14_val'],is_train=False,cfg=data_cfg)
val_loader = DataLoader(val_dataset,batch_size=cfg['training']['bs'],shuffle=False,pin_memory=True,num_workers=8)
print("Val Dataset length:",len(val_dataset))

if torch.cuda.is_available() and cfg['training']['cuda']:
    DEVICE='cuda'
else:
    DEVICE='cpu'
print("Device used:",DEVICE)

model_class = get_model_class(cfg=cfg)
print("Model class used:",model_class)

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
    print("Loading checkpoint from:",checkpoint)
    chkpt = torch.load(checkpoint)

    model.load_state_dict(chkpt['model_state_dict'])
    optimizer.load_state_dict(chkpt['optimizer_state_dict'])

    epochs_done = chkpt['epochs']

    train_loss = chkpt['train_loss']
    val_loss = chkpt['val_loss']
    best_val_loss = chkpt['best_val_loss']

    print("Epochs done:",epochs_done)
    print("Previous Train Loss:",train_loss)
    print("Previous Val Loss:",val_loss)
    print("Previous Best Val Loss:",best_val_loss)


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
            print("Epoch: {0}, {1}/{2}, Loss: {3}".format(ep,batch_num,len(train_loader),train_running_loss/batch_num),end=" ")
            for key in loss_dict:
                print("{0}: {1}".format(key,loss_dict[key]),end=" ")
            print()

    train_loss = train_running_loss/batch_num
    print("Epoch {0}: Train Loss: {1}".format(ep,train_loss))

    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch_num,batch in enumerate(val_loader,start=1):

            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(DEVICE)

            loss,loss_dict = model.validation_step(batch,criterion)

            val_running_loss += loss.detach().cpu()

            print("Val {0}/{1}, Loss: {2}".format(batch_num,len(val_loader),val_running_loss/batch_num),end=" ")
            for key in loss_dict:
                print("{0}: {1}".format(key,loss_dict[key]),end=" ")
            print(end="\r")
    
    val_loss = val_running_loss/batch_num 
    print()
    print("Epoch {0}: Val Loss: {1}".format(ep,val_loss))

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
        }

        if is_best:
            best_filename = os.path.join(curr_exp_dir,"best.pt")
            print("Saving best model at:",best_filename)
            torch.save(checkpoint_dict,best_filename)
        if ep%10 == 0:
            check_filename = os.path.join(curr_exp_dir,"check.pt")
            print("Saving every-10-epochs checkpoint model at:",check_filename)
            torch.save(checkpoint_dict,check_filename)


#Saving the last checkpoint
checkpoint_dict = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs": epochs_done+epochs,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "best_val_loss": best_val_loss,
}

last_filename = os.path.join(curr_exp_dir,"last.pt")
print("Saving last model at:",last_filename)
torch.save(checkpoint_dict,last_filename)

print("Training completed.")