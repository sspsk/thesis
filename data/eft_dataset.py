#packages imports
from torch.utils.data import Dataset
import json
import torch
import cv2
import numpy as np
from copy import deepcopy
from torchvision.transforms import Normalize

#local imports
from data.utils import augm_params,rgb_processing,pose_processing

class EFTDataset(Dataset):
    def __init__(self,annot_filenames,img_folders,dataset_len=-1,is_train=True):
        super().__init__()
        self.img_folders = img_folders
        self.dataset_len = dataset_len
        self.is_train = is_train
        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.annot_filenames = annot_filenames
        self.data = []

        for fname in self.annot_filenames:
            with open(fname,"r") as f:
                self.data.append(json.load(f)['data'])

        self.data_lens = torch.tensor([len(d) for d in self.data])
        
        
        
    def __len__(self):
        if self.dataset_len>0:
            return self.dataset_len
        else:
            return self.data_lens.sum()

    def __getitem__(self,idx):
        
        dataset_idx = (idx>=self.data_lens.cumsum(0)).sum()
        assert dataset_idx < len(self.data)
        sample_idx = idx - self.data_lens[:dataset_idx].sum()
        assert sample_idx < len(self.data[dataset_idx])
         
        idx_filename = self.data[dataset_idx][sample_idx]['imageName']

        scale = deepcopy(self.data[dataset_idx][sample_idx]['bbox_scale'])
        center = deepcopy(self.data[dataset_idx][sample_idx]['bbox_center'])

        flip, pn, rot, sc = augm_params(self.is_train)

        img = cv2.imread(self.img_folders[dataset_idx]+idx_filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)

        pose_params = np.array(deepcopy(self.data[dataset_idx][sample_idx]['parm_pose']))
        pose_params = pose_processing(pose_params,rot,flip,rotmat=True)#1 cause i have change the global rotation
        

        pose_params = torch.tensor(pose_params,dtype=torch.float32) 
        shape_params = torch.tensor(self.data[dataset_idx][sample_idx]['parm_shape'],dtype=torch.float32)
        cam_params = torch.tensor(self.data[dataset_idx][sample_idx]['parm_cam'],dtype=torch.float32)
        
        return img,(cam_params,pose_params,shape_params)