import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
import SimpleITK as sikt
from skimage.transform import  resize

class CustomImageDataset(Dataset):
    def __init__(self, CTImagePath, labelPath, imgTransform=None,labelTransform=None):
        self.CTImagePath = CTImagePath
        self.labelPath = labelPath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform

    def __len__(self):
        return len(self.CTImagePath)

    def __getitem__(self, idx):
        image = read_image(self.CTImagePath[idx]).astype(np.float32)
        label = read_label(self.labelPath[idx]).astype(np.int16)

        if self.imgTransform:
            image = self.imgTransform(image)
        if self.labelTransform:
            label = self.labelTransform(label)
        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(label)
        return image, label

def read_image(CTImagePath):
    img = sikt.ReadImage(CTImagePath)
    img = sikt.GetArrayFromImage(img)
    return  img
def read_label(labelPath):
    label = sikt.ReadImage(labelPath)
    label = sikt.GetArrayFromImage(label)
    return  label
def resizeFun(img,targetSize=(256,256,384)):
    return  resize(img,output_shape=targetSize,order=0,preserve_range=True)

if __name__ == '__main__':
    dataDirPath ="data/FLARE22_LabeledCase50-20220324T003930Z-001"
    imgPaths = list(map(lambda x: os.path.join(dataDirPath,"images",x),os.listdir(os.path.join(dataDirPath,"images"))))
    labelPath = list(map(lambda x: os.path.join(dataDirPath,"labels",x),os.listdir(os.path.join(dataDirPath,"labels"))))
    print(imgPaths)
    print(labelPath)
    splitIndex = int(len(imgPaths)*0.8)
    trainDataset = CustomImageDataset(CTImagePath=imgPaths[0:splitIndex],
                                      labelPath=labelPath[0:splitIndex],
                                      labelTransform=resizeFun,
                                      imgTransform=resizeFun)
    for img,label in trainDataset:
        print(img.size(),label.size())
        break
    print("total images:",len(trainDataset))


