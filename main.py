from exp_tracking.tracking_utils import begin_experiment,get_model_class
from data.eft_dataset import EFTDataset
import models
import config

import argparse
import torch
from torch.utils.data import DataLoader
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
args = parser.parse_args()

cfg,curr_exp_dir =  begin_experiment(args.config_file,force=args.force)

data_cfg = cfg['data']
datasets = data_cfg['datasets']

train_dataset = EFTDataset(datasets=datasets,is_train=True,cfg=data_cfg)
train_loader = DataLoader(train_dataset,batch_size=cfg['training']['bs'],shuffle=True,pin_memory=True)
print("Train Dataset length:",len(train_dataset))

val_dataset = EFTDataset(datasets=['coco14_val'],is_train=False,cfg=data_cfg)
val_loader = DataLoader(val_dataset,batch_size=cfg['training']['bs'],shuffle=False,pin_memory=True)
print("Val Dataset length:",len(val_dataset))

if torch.cuda.is_available() and cfg['training']['cuda']:
    DEVICE='cuda'
else:
    DEVICE='cpu'

model_class = get_model_class(cfg=cfg)
print("Model class used:",model_class)

model = model_class(cfg=cfg)
model = model.to(DEVICE)

optimizer = model.get_optimizer()
criterion = model.get_criterion()


checkpoint = cfg['training'].get("checkpoint",None)
if checkpoint is not None:
    pass
    #load model & optimizer


epochs = cfg['training']['epochs']
train_running_loss = 0.0
for ep in range(1,1+epochs):

    model.train()

    for batch_num,batch in enumerate(train_loader,start=1):

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        loss = model.train_step(batch,criterion)
        loss.backward()

        optimizer.step()

        train_running_loss += loss.detach().cpu()
        if batch_num % 10 == 0:
            print("Epoch: {0}, {1}/{2}, Loss: {3}".format(ep,batch_num,len(train_loader),train_running_loss/batch_num))


    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch_num,batch in enumerate(val_loader,start=1):

            batch = batch.to(DEVICE)

            loss = model.validation_step(batch,criterion)

            val_running_loss += loss.detach().cpu()

            print("Val {1}/{2}, Loss: {3}".format(batch_num,len(val_loader),val_running_loss/batch_num))
    
    
    #if best val loss, checkpointing
    #if epochs % 10 == 0 checkpointing




#TODO
#checkpoint loading
#training loop
#val loop