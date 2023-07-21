import numpy as np
import cv2
import torch
import sys
from torch.nn import functional as F

def load_mean_parameters(filename,rot6d=False,order="psc"):
    """
    Theta has shape (1,85) where:
    -[0,:3] -> 3 camera parameters
    -[0,3:75] -> pose parameters
    -[0,75:] -> shape parameters

    Note: From npz file returns always rot6d representation, even if rot6d is False
    """

    ext = filename.split(".")[-1]


    if ext == "h5":
        params = h5py.File(filename,'r')
    elif ext == "npz":
        params = np.load(filename)
    else:
        print("Unkown file extension. Exiting...")
        sys.exit()

    pose = torch.tensor(params['pose'],dtype=torch.float32)
    shape = torch.tensor(params['shape'],dtype=torch.float32)
    cam = torch.tensor([0.9,0.0,0.0],dtype=torch.float32)#the literature initializes the scale of camera with 0.9
    
    if ext == "h5":
        pose[:3] = torch.tensor([0,0,0],dtype=torch.float32)#original paper initializes global rotation as [pi/2,0,0]
    elif ext == "npz":
        pass
        #pose[:6] = torch.tensor([0,0,0,0,0,0],dtype=torch.float32)#original paper initializes global rotation as [pi/2,0,0]

    if rot6d and ext == "h5":
        pose = pose.unsqueeze(0)#shape [1,72]
        R = rodrigues_formula(pose)# shape[1,24,3,3]
        a1 = R[:,:,:,0]
        a2 = R[:,:,:,1]
        pose = torch.concat([a1,a2],axis=-1).squeeze(0).flatten()

    
    if order=="cps":
        theta = torch.concat([cam,pose,shape])
    elif order == "psc":
        theta = torch.concat([pose,shape,cam])
    else:
        print("Invalid (cam,pose,shape) order given, must be 'cps' or 'psc'. Exiting...")
        sys.exit()
        
    return theta.unsqueeze(0)

def augm_params(is_train):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling

        noise_factor = 0.4
        rot_factor=10
        scale_factor=0.25

        if is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-noise_factor, 1+noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*rot_factor,
                    max(-2*rot_factor, np.random.randn()*rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+scale_factor,
                    max(1-scale_factor, np.random.randn()*scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def crop(img,center,scale,res,rot=0):
    bbox_size = 200*scale

    new_rows = res[0]
    new_cols = res[1]

    center = np.array([[center[0],center[1]]])
    new_center = np.array([[new_cols//2,new_rows//2]])

    theta=np.deg2rad(rot)
    R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    dir_down = np.array([[0.0,bbox_size*0.5]])
    dir_down = (R @ dir_down.reshape(2,1)).reshape(1,2)


    dir_right = np.array([[bbox_size*0.5,0.0]])
    dir_right = (R@ dir_right.reshape(2,1)).reshape(1,2)

    new_down = np.array([[0.0,new_rows*0.5]])
    new_right = np.array([[new_cols*0.5,0.0]])

    pts1 = np.concatenate([center,center+dir_down,center+dir_right]).astype(np.float32)
    pts2 = np.concatenate([new_center,new_center+new_down,new_center+new_right]).astype(np.float32)

    M = cv2.getAffineTransform(pts1,pts2)

    new_img_cropped = cv2.warpAffine(img,M,(new_cols,new_rows)).astype(np.float32)

    return new_img_cropped

def rgb_processing(rgb_img,center,scale,rot,flip,pn):
    IMG_RES=224
    rgb_img = crop(rgb_img,center,scale,[IMG_RES,IMG_RES],rot=rot)

    if flip:
        rgb_img = flip_img(rgb_img)

    rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
    rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
    rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))

    #! permute and normalize
    rgb_img = np.transpose(rgb_img,(2,0,1))/255.0
    return rgb_img

def flip_pose(pose,rotmat=False):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    if rotmat:
        pose = pose[SMPL_JOINTS_FLIP_PERM] 
        pose[:,1,0] *= -1
        pose[:,2,0] *= -1
        #pose[:,0,0] *= -1
        pose[:,0,1] *= -1
        pose[:,0,2] *= -1
    else:
        SMPL_POSE_FLIP_PERM = []
        for i in SMPL_JOINTS_FLIP_PERM:
            SMPL_POSE_FLIP_PERM.append(3*i)
            SMPL_POSE_FLIP_PERM.append(3*i+1)
            SMPL_POSE_FLIP_PERM.append(3*i+2)
        flipped_parts = SMPL_POSE_FLIP_PERM
        pose = pose[flipped_parts]
        # we also negate the second and the third dimension of the axis-angle
        pose[1::3] = -pose[1::3]
        pose[2::3] = -pose[2::3]
    return pose

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa
    
def pose_processing(pose, r, f, rotmat=False):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        if rotmat:
            theta = -np.deg2rad(r)#minus because i use the rotation matrix formula for the positive z axis
            R = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
            pose[0,:,:] = R @ pose[0,:,:]
        else:
            pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose,rotmat=rotmat)
        # (72),float
        pose = pose.astype('float32')
        return pose

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)