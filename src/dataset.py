import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])


class SegformerDataset(Dataset):
    def __init__(self, root_dir:str, mode:str):
        """Dataset class for Cityscapes semantic segmentation data
        Args:
            rootDir (str): path to directory containing cityscapes image data
            folder (str) : 'train' or 'val' folder
        """        
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

        # read rgb image list
        sourceImgFolder =  os.path.join(self.root_dir, 'images', self.mode)
        self.sourceImgFiles  = [os.path.join(sourceImgFolder, x) for x in sorted(os.listdir(sourceImgFolder))]

        # read label image list
        seglabelImgFolder =  os.path.join(self.root_dir, 'seg_labels', self.mode)
        self.seglabelImgFiles  = [os.path.join(seglabelImgFolder, x) for x in sorted(os.listdir(seglabelImgFolder))]

        depthlabelImgFolder =  os.path.join(self.root_dir, 'depth_labels', self.mode)
        self.depthlabelImgFiles  = [os.path.join(depthlabelImgFolder, x) for x in sorted(os.listdir(depthlabelImgFolder))]
    
    def __len__(self):
        return len(self.sourceImgFiles)
  
    def __getitem__(self, index):
        # read source image and convert to RGB, apply transform
        sourceImage = np.load(self.sourceImgFiles[index])
        if self.transform is not None:
            sourceImage = self.transform(sourceImage)
            sourceImage = sourceImage.float()

        # read label image and convert to torch tensor
        seglabelImage = np.load(self.seglabelImgFiles[index])
        seglabelImage = torch.from_numpy(seglabelImage).long()

        depthlabelImage = 10.0 * np.load(self.depthlabelImgFiles[index])
        depthlabelImage = torch.from_numpy(depthlabelImage).permute(2,0,1).float()
        return sourceImage, seglabelImage, depthlabelImage        


###################################
# FUNCTION TO GET TORCH DATASET  #
###################################

def get_cs_datasets(rootDir):
    data = SegformerDataset(rootDir, folder='train', tf=preprocess)
    test_set = SegformerDataset(rootDir, folder='val', tf=preprocess)

    # split train data into train, validation and test sets
    total_count = len(data)
    train_count = int(0.8 * total_count) 
    train_set, val_set = torch.utils.data.random_split(data, (train_count, total_count - train_count), 
            generator=torch.Generator().manual_seed(1))
    return train_set, val_set, test_set

def get_dataloaders(train_set, val_set, test_set):
    train_dataloader = DataLoader(train_set, batch_size=32,drop_last=True)
    val_dataloader   = DataLoader(val_set, batch_size=8)
    test_dataloader  = DataLoader(test_set, batch_size=8)
    return train_dataloader, val_dataloader, test_dataloader 