#package imports
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torch
import os
import json
import argparse


#local imports
from data.eval_dataset import Dataset_3DPW
import config
from data.utils import rot6d_to_rotmat,reconstruction_error
from exp_tracking.tracking_utils import get_checkpoint_path, get_model_class,parse_config
import models


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',type=str,default='exp_config.yml')
parser.add_argument('--force',action='store_true',help='Force to begin experiment with uncommited changes')
args = parser.parse_args()

cfg = parse_config(args.config_file)

eval_dataset = Dataset_3DPW()
eval_loader = DataLoader(eval_dataset,batch_size=cfg['training']['bs'],shuffle=False,num_workers=8)
print("Dataset len:",len(eval_dataset))

checkpoint_path = get_checkpoint_path = get_checkpoint_path(cfg)
model_class = get_model_class(cfg)

model = model_class(cfg=cfg)


if checkpoint_path is not None:
    print("Loading checkpoint from:",checkpoint_path)
    chkpt = torch.load(checkpoint_path)
    model.load_state_dict(chkpt['model_state_dict'])
    epochs = chkpt['epochs']

if epochs is not None:
    print("Epochs:",epochs)

if torch.cuda.is_available() and cfg['training']['cuda']:
    DEVICE='cuda'
else:
    DEVICE='cpu'
print("Device used:",DEVICE)
        

model = model.to(DEVICE)
model.eval()

rec_error = []

with torch.no_grad():
    for bn,batch in enumerate(eval_loader,start=1):

        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(DEVICE)

        loss = model.eval_step(batch)
        #rec_error.append(reconstruction_error(Jtr_pred.cpu().numpy(),Jtr_gt.cpu().numpy(),reduction='sum'))
        rec_error.append(loss)
        print("Batch: {0}/{1}".format(bn,len(eval_loader)),end='\r')
        


print()
print("Reconstruction error:",sum(rec_error)/len(eval_dataset)) 

results_path = os.path.join(cfg['metadata']['exp_dir'],cfg['metadata']['name'],'results.json')
with open(results_path,'w') as f:
    json.dump({'pampjpe':sum(rec_error)/len(eval_dataset)},f)

print("Results saved at:",results_path)