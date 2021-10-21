""" 추론 코드

TODO:

NOTES:

REFERENCE:
    * MNC 코드 템플릿 predict.py

UPDATED:
"""
import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from modules.dataset import TestDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_csv
import torch
from model.model import PestClassifier
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from modules.metrics import get_metric_fn
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# # CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = '../shared/hackathon/Raw/test'

PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yml')
config = load_yaml(PREDICT_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# PREDICT
BATCH_SIZE = config['PREDICT']['batch_size']
### Input Shape 설정 필요 ###
INPUT_SHAPE = (128, 128)


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    ### Best Model, pred 저장 위치 설정 필요 ###
    TRAINED_MODEL_PATH = 'results/train/Efficientb6_20211015102612/best.pt'
    SAVE_PATH = 'results/csv/pred4.csv'
    ######

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TestDataset(data_dir=DATA_DIR, input_shape=INPUT_SHAPE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print('Test set samples:',len(test_dataset))

    criterion = nn.CrossEntropyLoss()
    metric_fn = get_metric_fn


    # Load Model
    model = PestClassifier(num_class=10).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))['model'])
    # model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu'))['model'])

    pred_lst = []
    file_name_lst = []

    with torch.no_grad():
        for batch_index, (img, file_name) in enumerate(test_dataloader):
            img = img.to(device)
            pred = model(img)
            pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
            file_name_lst.extend(file_name)
            
    df = pd.DataFrame({'file_name':file_name_lst, 'answer':pred_lst})
    df.to_csv(SAVE_PATH, index=None)


