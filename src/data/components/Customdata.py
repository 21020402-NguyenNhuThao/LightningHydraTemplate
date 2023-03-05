import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
)
from PIL import Image, ImageDraw
import scipy.io
import glob
import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple
import albumentations as A
import albumentations.pytorch.transforms as T

def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))
class customData(Dataset):
    img_root = "ibug_300W_large_face_landmark_dataset"

    def __init__(self,
                 root: str = '',
                 img_dims: Tuple[int, int] = [3, 64, 64],
                 augmentation: bool = False) -> None:
        img_dir =  [
            f"{self.img_root}/afw/*[0-9].jpg",
            f"{self.img_root}/helen/trainset/*[0-9].jpg",
            f"{self.img_root}/helen/testset/*[0-9].jpg",
            f"{self.img_root}/ibug/*[0-9].jpg",
            f"{self.img_root}/lfpw/trainset/*[0-9].jpg",
            f"{self.img_root}/lfpw/testset/*[0-9].jpg"
        ]
        keypoint_dir = [
            f"{self.img_root}/afw/*.pts",
            f"{self.img_root}/helen/trainset/*.pts",
            f"{self.img_root}/helen/testset/*.pts",
            f"{self.img_root}/ibug/*.pts",
            f"{self.img_root}/lfpw/trainset/*.pts",
            f"{self.img_root}/lfpw/testset/*.pts"
        ]
        bb_paths = [
            "bounding_boxes_afw.mat", # 337

            "bounding_boxes_helen_trainset.mat", # 2000
            "bounding_boxes_helen_testset.mat", # 330

            #  Dont know why ibug link but has afw bb?
            "bounding_boxes_ibug.mat", # 337

            "bounding_boxes_lfpw_trainset.mat", # 811
            "bounding_boxes_lfpw_testset.mat", # 224

            # f"{self.dataset_dir}/Bounding Boxes/bounding_boxes_xm2vts.mat",
        ]
        self.bb_paths = sorted(glob.glob(bb_paths[0]) + glob.glob(bb_paths[1])
                                + glob.glob(bb_paths[2]) + glob.glob(bb_paths[3]) 
                                + glob.glob(bb_paths[4]) + glob.glob(bb_paths[5]) )
        transforms = [Resize((224,224))]

        self.transform = Compose(transforms)
        self.augument = None
        self.img_paths = sorted(glob.glob(img_dir[0]) + glob.glob(img_dir[1])
                                + glob.glob(img_dir[2]) + glob.glob(img_dir[3]) 
                                + glob.glob(img_dir[4]) + glob.glob(img_dir[5]) )
        self.keypoint_paths = sorted(glob.glob(keypoint_dir[0]) + glob.glob(keypoint_dir[1])
                                + glob.glob(keypoint_dir[2]) + glob.glob(keypoint_dir[3]) 
                                + glob.glob(keypoint_dir[4]) + glob.glob(keypoint_dir[5]) )
       
        super().__init__()

    def __len__(self) -> int:
        return len(self.img_paths)
    
    
    
    def __getitem__(self, idx) -> torch.Tensor:
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        

        kp = read_pts(self.keypoint_paths[idx]).tolist()
        bb = self.load_data()
        box = (bb[idx][0][0][2][0][0], bb[idx][0][0][2][0][1],
                bb[idx][0][0][2][0][2], bb[idx][0][0][2][0][3])
        box = np.array(box, dtype=float)

        self.transform = A.Compose([
          
           A.Crop(x_min=int(box[0]),y_min=int(box[1]),x_max=int(box[2]),y_max=int(box[3])),
           A.augmentations.geometric.resize.Resize(height=224,width=224),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
            A.HueSaturationValue(p=0.5), 
            A.RGBShift(p=0.7)
        ], p=1),                          
            A.RandomBrightnessContrast(p=0.5)

        ], keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))
        if self.transform:
            sample = self.transform(image =  image, keypoints = kp)
            width , height = 224,224
            sample["keypoints"] = sample["keypoints"]/np.array([width, height]) - 0.5
        
        return sample
    
    def load_data(self, path=None, boxes=None):
        boxes = [] # boxes with 6 links above
        for path in self.bb_paths:
          box = scipy.io.loadmat(path)
          boxes.append(box)
          bbs = []
          for sublist in boxes:
            for b in sublist['bounding_boxes']:
              for bb in b:
                bbs.append(bb)      
        return bbs
    



        
    
    
        