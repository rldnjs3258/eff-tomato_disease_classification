# eff-tomato_disease_classification
 - writer : Kiwon Seo

<br>
<hr>
<br>

## 1. How to use
 - run : python train.py
 - predict : python predict.py
 - ensemble : python ensemble.py
 
<br>
<hr>
<br>

## 2. 확장 - 플랫폼
### (1) WanDB란?
 - WanDB는 **Weight & Biases**의 약자로, 모델 개발을 위한 툴이다.
 - WanDB는 Tensorflow, Pytorch 등 **여러 프레임워크**에서 사용 가능 하다.
 - WanDB를 통해 **MLOps**로의 확장이 가능 하다.
 - WanDB는 **모델 모니터링, 모델 실험, Artifacts, 대시보드** 등의 기능을 지원 한다.   
 
    모델 모니터링 : 실시간으로 모델 loss, accuracy 등 모니터링    
    
    모델 실험 : 모델, config, 하이퍼 파라미터 튜닝 등 실험 및 저장
    
    Artifacts : 데이터 셋 버전 관리, 모델 버전 관리    
    
    대시보드 : GPU 사용량, Loss, Accuracy 등 대시보드 형태로 확인 가능

<br>

### (2) WanDB 적용 내용
 - Refference : https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
 - ID : 깃헙 계정
 - Full name : Ben Seo   
 
    Username : benseo
    
    Organization : Kyowon
    
    Project Name : eff-tomato_disease_classification
    
    Project : https://wandb.ai/benseo/eff-tomato_disease_classification
    
    Project Run : https://wandb.ai/benseo/eff-tomato_disease_classification/runs/실험-이름

<br>

### (3) WanDB 코드 스니펫 (튜토리얼)
 - Train 파일에 아래의 코드를 적용 해서 모델을 추적 하면 됨!
```python
import wandb

# 1. Start a new run
wandb.init(project='eff-tomato_disease_classification', entity='benseo')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# 3. Log gradients and model parameters
wandb.watch(model)
for batch_idx, (data, target) in enumerate(train_loader):
  ...
  if batch_idx % args.log_interval == 0:
    # 4. Log metrics to visualize performance
    wandb.log({"loss": loss})
```

<br>

### (4) WanDB 코드 실제 적용
#### 1) train.py 적용 내용
```python
# WandB : 라이브러리 로드
import wandb

### ...

# WandB : wandb 세팅
wandb.init(project='eff-tomato_disease_classification', entity='benseo', config={"num_epochs": config['TRAIN']['num_epochs'], "batch_size": config['TRAIN']['batch_size'], "learning_rate": config['TRAIN']['learning_rate'], "early_stopping_patience": config['TRAIN']['early_stopping_patience'], "model": config['TRAIN']['model'], "input_shape": config['TRAIN']['input_shape'], "layer": config['TRAIN']['layer'], "img_aug": config['TRAIN']['img_aug'], "softmax": config['TRAIN']['softmax'], "initialization": config['TRAIN']['initialization']}) # 실험 init 설정
wandb.run.name = config['TRAIN']['model'] + '-layer(' + config['TRAIN']['layer'] + ')-img_aug(' + config['TRAIN']['img_aug'] + ')-softmax(' + config['TRAIN']['softmax'] + ')-' + config['TRAIN']['initialization'] # 실험 이름 설정
wandb.run.save()

### ...

# WandB : 모델 gradient 추적
wandb.watch(model) # 모델 gradient 추적

### ...

# WandB : train loss, validation loss, train score, validation score 추적
wandb.log({"train_loss": trainer.train_mean_loss, "validation_loss": trainer.val_mean_loss, "train_score": trainer.train_score, "validation_score": trainer.validation_score})
```

<br>

#### 2) trainer.py 적용 내용
```python
# WandB : 라이브러리 로드
import wandb

### ...

# WandB : Validation Image 리스트
val_images_lst = []
with torch.no_grad():
    for batch_index, (img, label) in enumerate(tqdm(dataloader)):
        # print("[     {0} / {1}      ]".format(batch_index,len(dataloader)),end="\r")
        img = img.to(self.device)
        label = label.to(self.device).long()
        pred = self.model(img)
        loss = self.criterion(pred, label)
        val_total_loss += loss.item()
        prob_lst.extend(pred[:, 1].cpu().tolist())
        target_lst.extend(label.cpu().tolist())
        pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
        # WandB : Validation Image 리스트에 이미지, pred, label 저장
        val_images_lst.append(wandb.Image(img, caption="Pred: {} Truth: {}".format(pred.argmax(dim=1), label)))
    self.val_mean_loss = val_total_loss / batch_index
    self.validation_score = self.metric_fn(y_pred=pred_lst, y_answer=target_lst, y_prob=prob_lst)
    msg = f'Epoch {epoch_index}, {mode} loss: {self.val_mean_loss}, Acc: {self.validation_score}'
    print(msg)
    # WandB : 로그 출력
    wandb.log({"val images": val_images_lst})
```

<br>
<hr>
<br>

## 3. 성능 개선 및 실험
### (1) Baseline
 - Pretrained Model : EfficientNet-b2
 - Layer : None
 - Image Augmentation : None
 - Early Stopping Patience : 20
 - Softmax : True
 - Initialization : Xavier
 - Batch Size : 128
 - Input Shape : (128, 128)
 - Val Loss : 
 - Val Accuracy : 
 
<br>
<hr>
<br>

### (2) 성능 개선
#### 1) EfficientNetb6 적용 및 레이어 쌓기
 - Pretrained Model : EfficientNetb6
 - 레이어 : 1280 -> 500 -> 250 -> 10
```python
class PestClassifier(nn.Module):
    def __init__(self, num_class):
        super(PestClassifier, self).__init__()
        # Pretrained Model : efficientnet-b3
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1280)

        # Top Layer : effificientnet-b3에 fully-connected 레이어를 쌓아서 최종 레이어에서는 10개가 output (클래스가 10개임)
        # Pretrained model 이후 쌓는 레이어들로, 실험적 혹은 경험적 결과로 레이어를 쌓으면 됨
        num_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Linear(num_features, 500),
                                 nn.BatchNorm1d(500),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(500, 250),
                                 nn.BatchNorm1d(250),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(250, num_class))

# pytorch는 loss를 crossentropy로 하면 softmax가 같이 실행 됨
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
#         # Softmax -> logsoftmax로 바꾸기
#         self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_img):
        # Model
        x = self.model(input_img)
        
#         # Softmax -> logsoftmax로 바꾸기
#         x = self.softmax(x)
        
        return x
```

<br>

#### 2) Image Augmentation 적용
```python
        self.transform = transforms.Compose([
            transforms.Resize(self.input_shape), # default
            transforms.RandomRotation(90), # 이미지 랜덤 회전
            transforms.RandomHorizontalFlip(p=0.5), # 이미지 랜덤 수평 뒤집기
            transforms.RandomVerticalFlip(p=0.5), # 이미지 랜덤 수직 뒤집기
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # color 변환은 성능을 저하 시킬 수도 있음
            transforms.ToTensor(), # default
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # default
        ])
```

<br>

#### 3) Early Stopping 없애는 것 고려 하기
 - 이유 : 진동이 많을 경우 local minimum에 수렴 할 수 있음 (https://medium.com/@codecompose/resnet-e3097d2cfe42)
 - Epochs을 50으로 두고 early_stopping_patience를 50으로 둬서 없는 것과 같이 이용
```
TRAIN:
  num_epochs: 50
  batch_size: 128 # 128 -> 256
  learning_rate: 0.0005
  early_stopping_patience: 50 # early stop 쓸지 고려 (진동 https://medium.com/@codecompose/resnet-e3097d2cfe42)
  model: Efficientb6
  optimizer:
  scheduler:
  momentum:
  weight_decay: 0.00001
  loss_function:
  metric_function:
  input_shape: 128 # Efficientnet의 중요 파라미터
  layer: '1000-512-256-10'
  img_aug: 'True'
  softmax: 'N'
  initialization : 'Xavier'
```

<br>

#### 4) Softmax 떼기
 - 이유 : Pytorch의 Crossentropy loss에는 softmax가 이미 붙어 있음. 모델에 softmax를 붙이면 softmax를 두 번 이용 하는 셈임

<br>

#### 5) Initializer 설정
 - Default Initializer인 xavier에서 케이밍으로 변경 하기

<br>

#### 6) Input Size를 권장 사이즈로 하기

<br>
<hr>
<br>

### (3) 추가 실험
 - Majority Voting Ensemble 실험 (추후 클래스 vote가 아닌 확률 값 mean으로 앙상블)

<br>
<hr>
<br>

## 4. 기타
 - 블로그 : 추후 블로그 화 하기
 - Colab GPU or TPU로 학습 하기 (TPU로 이용 할 경우 데이터셋 변환이 필요 하나 속도 뿐 아니라 계산 성능도 좋아짐. float32 -> float64를 계산 할 경우 소숫점 밑 64자리를 더 정확히 계산하게 됨)
 - Test set에도 Image Augmentation 적용 하는 TTD 실험 하기
 - Pytorch Lightening 적용 하기 (https://www.pytorchlightning.ai/tutorials) 
 - streamlit
 - 주석 달기