#packages imports
from torch.utils.data import Dataset
import json
import torch
import cv2
import numpy as np
from copy import deepcopy
from torchvision.transforms import Normalize

#local imports
from data.utils import augm_params,rgb_processing,pose_processing,kp_processing,kp3d_processing
import config
import constants

eft_annot_files = {
    'mpii':config.MPII_ANNOT_FILE,
    'coco14':config.COCO14_ANNOT_FILE,
    'coco14_val': config.COCO14_VAL_ANNOT_FILE,
    'mpiinf':config.MPIINF_EFT_ANNOT_FILE

}

eft_img_folders= {
    'mpii': config.MPII_IMAGES_FOLDER,
    'coco14': config.COCO14_IMAGES_FOLDER,
    'coco14_val': config.COCO14_VAL_IMAGES_FOLDER,
    'mpiinf': config.MPIINF_IMAGE_FOLDER
}

class EFTDataset(Dataset):
    def __init__(self,datasets=['coco14'],is_train=True,cfg={}):
        super().__init__()


        self.annot_filenames = [eft_annot_files[d] for d in datasets]
        self.img_folders = [eft_img_folders[d] for d in datasets]
        self.cfg = cfg
        self.dataset_len = cfg.get('dataset_length',None)
        self.is_train = is_train
        self.normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data = []

        for fname in self.annot_filenames:
            with open(fname,"r") as f:
                self.data.append(json.load(f)['data'])

        self.data_lens = torch.tensor([len(d) for d in self.data])
        
        
        
    def __len__(self):
        if self.dataset_len is not None:
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

        img,M = rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)

        keypoints2d = np.array(self.data[dataset_idx][sample_idx]['gt_keypoint_2d'])
        visibility2d = torch.from_numpy(keypoints2d[:,2]).to(torch.float32)
        if flip:
            if visibility2d.shape[0] == 49:
                visibility2d = visibility2d[constants.J49_FLIP_PERM]
            else:
                visibility2d = visibility2d[constants.J24_FLIP_PERM]
        keypoints2d = torch.from_numpy(kp_processing(keypoints2d[:,:2],flip,M)).to(torch.float32)

        if 'gt_keypoint_3d' in self.data[dataset_idx][sample_idx]:
            keypoints3d = np.array(self.data[dataset_idx][sample_idx]['gt_keypoint_3d'])
            visibility3d = torch.from_numpy(keypoints3d[:,3]).to(torch.float32)
            if flip:
                visibility3d = visibility3d[constants.J24_FLIP_PERM]
            keypoints3d = torch.from_numpy(kp3d_processing(keypoints3d[:,:3],flip,rot)).to(torch.float32)
        else:
            keypoints3d = torch.zeros(24,3)
            visibility3d = torch.zeros(2,1)

        has_smpl = self.data[dataset_idx][sample_idx].get('has_smpl',0.0)

        pose_params = np.array(deepcopy(self.data[dataset_idx][sample_idx]['parm_pose']))
        pose_params = pose_processing(pose_params,rot,flip,rotmat=True)

        pose_params = torch.tensor(pose_params,dtype=torch.float32) 
        shape_params = torch.tensor(self.data[dataset_idx][sample_idx]['parm_shape'],dtype=torch.float32)
        cam_params = torch.tensor(self.data[dataset_idx][sample_idx]['parm_cam'],dtype=torch.float32)
        
        ret_dict = {}
        ret_dict['img']=img
        ret_dict['cam']=cam_params
        ret_dict['pose']=pose_params
        ret_dict['shape']=shape_params
        ret_dict['keypoints2d'] = keypoints2d
        ret_dict['visibility2d'] = visibility2d
        ret_dict['keypoints3d'] = keypoints3d
        ret_dict['visibility3d'] = visibility3d
        ret_dict['has_smpl'] = has_smpl

        return  ret_dict