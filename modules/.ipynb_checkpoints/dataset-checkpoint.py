"""Dataset 클래스 정의
TODO:
NOTES:
UPDATED:
"""

import os
import copy
import cv2
import torch
import sys
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode, input_shape):
        self.data_dir = os.path.join(data_dir, mode)
        self.mode = mode
        self.input_shape = input_shape
        self.db = self.data_loader()
        # Image Augmentation (Efficientnet은 모델 버전에 따라 Normalize를 주의 해서 해야 함)
        # reference : https://colab.research.google.com/github/sanchit2843/MLBasics/blob/master/IntelClassificationKaggle/Pytorch%20transfer%20learning%20tutorial%20%5B93%25acc%5D.ipynb
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape), 
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.RandomCrop(20),
            ])
#        self.transform = transforms.Compose([transforms.Resize(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.class_num = len(self.db['label'].unique())
        
    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()
        class_list = {'Tomato_D01':0, 'Tomato_D04':1, 'Tomato_D05':2, 'Tomato_D07':3, 'Tomato_D08':4, 'Tomato_D09':5, 'Tomato_H':6, 'Tomato_P03':7, 'Tomato_P05':8, 'Tomato_R01':9}

        image_path_list = []
        image_label_list = []
        for (root, dirs, files) in os.walk(self.data_dir):
            if root.split('/')[-1] in class_list.keys():
                label = class_list[root.split('/')[-1]]
            else:
                continue
            for filename in files:
                if filename.split('.')[-1] == 'png':
                    image_path_list.append(os.path.join(root, filename))
                    image_label_list.append(label)
                else:
                    print(filename)
        db = pd.DataFrame({'img_path': image_path_list, 'label': image_label_list})
        return db

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.db.loc[index])

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        # 2. preprocessing images
        trans_image = self.transform(Image.fromarray(cvimg))

        return trans_image, data['label']

class TestDataset(Dataset):
    def __init__(self, data_dir, input_shape):
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.db = self.data_loader()
        # Image Augmentation은 Train Dataset에만 적용
        self.transform = transforms.Compose([transforms.Resize(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def data_loader(self):
        print('Loading test dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        image_path_list = []
        image_label_list = []
        x_size_list = []
        y_size_list = []

        for (path, dirs, files) in os.walk(self.data_dir):
            for filename in [f for f in files if 'checkpoint' not in f]:
                ext = os.path.splitext(filename)[-1]
                if ext == '.png':
                    image_path_list.append(os.path.join(path, filename))
        
        image_path_list = sorted(image_path_list, key=lambda x : int(x.split('/')[-1].split('.')[0]))
        db = pd.DataFrame({'img_path': image_path_list})
        return db
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, index):
        data = copy.deepcopy(self.db.loc[index])
        
         #1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])

        # 2. preprocessing images
        trans_image = self.transform(Image.fromarray(cvimg))
        return trans_image, data['img_path'].split('/')[-1]