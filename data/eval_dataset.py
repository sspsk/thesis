from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from torchvision import transforms as T
import torch
import numpy as np
import cv2
from copy import deepcopy
import json

from data.utils import crop
import config

class Dataset_3DPW(Dataset):
    def __init__(self,ds_len=None):
        self.data = []
        self.annot_file = config.PW3D_ANNOT_FILE
        self.img_folder = config.PW3D_IMAGE_FOLDER
        self.transforms = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ds_len = ds_len

        with open(self.annot_file,"r") as f:
            self.data = json.load(f)

        if self.ds_len is not None:
            self.data = self.data[:ds_len]
            

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self,idx):
        idx_filename = self.data[idx]['imgname']
        scale = self.data[idx]['scale']
        center = self.data[idx]['center']

        img = cv2.imread(self.img_folder+idx_filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = crop(img,center,scale,res=(224,224),rot=0).astype(np.uint8)

        img = self.transforms(img)

        pose= deepcopy(self.data[idx]['pose'])
        pose = np.array(pose)
    
        pose_params = torch.tensor(pose,dtype=torch.float32)
        shape_params = torch.tensor(self.data[idx]['shape'],dtype=torch.float32)
        
        return dict(img=img,pose=pose_params,shape=shape_params)
