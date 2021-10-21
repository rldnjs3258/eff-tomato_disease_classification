""" 학습 코드

TODO:

NOTES:

REFERENCE:
    * MNC 코드 템플릿 train.py

UPDATED:
"""

# WandB : 라이브러리 로드
import wandb
import os
import random
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime, timezone, timedelta
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.metrics import get_metric_fn
from modules.dataset import CustomDataset , TestDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
import torch
from model.model import PestClassifier

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = '../shared/hackathon/Split/'
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config_128.yml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
MODEL = config['TRAIN']['model']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']
# efficientnet 모델 버전에 따라 INPUT SHAPE 아주 중요함!
INPUT_SHAPE = config['TRAIN']['input_shape']
INPUT_SHAPE = (INPUT_SHAPE, INPUT_SHAPE)

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{MODEL}_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

# TRAIN CONFIG LIST
def train_config(train_config_list):
    TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, train_config_list)
    config = load_yaml(TRAIN_CONFIG_PATH)

    INPUT_SHAPE = config['TRAIN']['input_shape']
    INPUT_SHAPE = (INPUT_SHAPE, INPUT_SHAPE)
    
    return INPUT_SHAPE

if __name__ == '__main__':    
    # config 설정에 따라 3개의 config 하이퍼 파라미터 실험 (input_shape만 바꿈)
    train_config_list = ['config/train_config_128.yml']
    
    for i in train_config_list:
        INPUT_SHAPE = train_config(i)
        
        # WandB : wandb 세팅
        wandb.init(project='eff-tomato_disease_classification', entity='benseo', config={"num_epochs": config['TRAIN']['num_epochs'], "batch_size": config['TRAIN']['batch_size'], "learning_rate": config['TRAIN']['learning_rate'], "early_stopping_patience": config['TRAIN']['early_stopping_patience'], "model": config['TRAIN']['model'], "input_shape": config['TRAIN']['input_shape'], "layer": config['TRAIN']['layer'], "img_aug": config['TRAIN']['img_aug'], "softmax": config['TRAIN']['softmax'], "initialization": config['TRAIN']['initialization']}) # 실험 init 설정
        wandb.run.name = config['TRAIN']['model'] + '-layer(' + config['TRAIN']['layer'] + ')-early_stopping(' + str(config['TRAIN']['early_stopping_patience']) + ')-img_aug(' + config['TRAIN']['img_aug'] + ')-softmax(' + config['TRAIN']['softmax'] + ')-' + config['TRAIN']['initialization'] # 실험 이름 설정
        wandb.run.save()
    
        # Set random seed
        torch.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set train result directory
        make_directory(PERFORMANCE_RECORD_DIR)

        # Set system logger
        system_logger = get_logger(name='train', file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

        # Load dataset & dataloader
        train_dataset = CustomDataset(data_dir=DATA_DIR, mode='train', input_shape=INPUT_SHAPE)
        validation_dataset = CustomDataset(data_dir=DATA_DIR, mode='val', input_shape=INPUT_SHAPE)
    #     test_dataset = CustomDataset(data_dir=DATA_DIR, mode='test', input_shape=INPUT_SHAPE)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #     test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #   print('Train set samples:',len(train_dataset),  'Val set samples:', len(validation_dataset), 'Test set samples:', len(test_dataset))
        print('Train set samples:',len(train_dataset))
        # import pdb;pdb.set_trace()
        # size 128 128 3

        # Load Model
        model = PestClassifier(num_class=train_dataset.class_num).to(device)
        
        # WandB : 모델 gradient 추적
        wandb.watch(model) # 모델 gradient 추적

        # Set optimizer, scheduler, loss function, metric function
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # 원 사이클 돌려서 lr을 찾음
        scheduler =  optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=EPOCHS, steps_per_epoch=len(train_dataloader))
        criterion = nn.CrossEntropyLoss() # pytorch의 crossentropyloss를 쓰면 softmax를 안 써도 됨
        metric_fn = get_metric_fn

        # Set trainer
        trainer = Trainer(criterion, model, device, metric_fn, optimizer, scheduler, logger=system_logger)

        # Set earlystopper
        early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

        # Set performance recorder
        key_column_value_list = [
            TRAIN_SERIAL,
            TRAIN_TIMESTAMP,
            MODEL,
            OPTIMIZER,
            LOSS_FN,
            METRIC_FN,
            EARLY_STOPPING_PATIENCE,
            BATCH_SIZE,
            EPOCHS,
            LEARNING_RATE,
            WEIGHT_DECAY,
            RANDOM_SEED]

        performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                                   record_dir=PERFORMANCE_RECORD_DIR,
                                                   key_column_value_list=key_column_value_list,
                                                   logger=system_logger,
                                                   model=model,
                                                   optimizer=optimizer,
                                                   scheduler=scheduler)

        # Train
        save_yaml(os.path.join(PERFORMANCE_RECORD_DIR, 'train_config.yaml'), config)
        criterion = 1E+8
        for epoch_index in tqdm(range(EPOCHS)):
            trainer.train_epoch(train_dataloader, epoch_index)
            trainer.validate_epoch(validation_dataloader, epoch_index, 'val')

            # Performance record - csv & save elapsed_time
            performance_recorder.add_row(epoch_index=epoch_index,
                                         train_loss=trainer.train_mean_loss,
                                         validation_loss=trainer.val_mean_loss,
                                         train_score=trainer.train_score,
                                         validation_score=trainer.validation_score)
            
            # WandB : train loss, train score, validation loss, validation score 추적
            wandb.log({"train_loss":trainer.train_mean_loss, "validation_loss": trainer.val_mean_loss, "train_score":trainer.train_score, "validation_score": trainer.validation_score})
            
            # Performance record - plot
            performance_recorder.save_performance_plot(final_epoch=epoch_index)

            # early_stopping check
            early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

            if early_stopper.stop:
                print('Early stopped')
                break
                print("Val Mean Loss : ",trainer.val_mean_loss)
            # print("Criterion : ",   criterion)
            # import pdb;pdb.set_trace()
            # if trainer.val_mean_loss < criterion:
            
            # train loss -> val loss로 해야 early stop이 걸림
            if trainer.val_mean_loss < criterion:
                criterion = trainer.val_mean_loss
                performance_recorder.weight_path = os.path.join(PERFORMANCE_RECORD_DIR, 'best.pt')
                performance_recorder.save_weight()
                print(f'{epoch_index} model saved')
                print('----------------------------------')