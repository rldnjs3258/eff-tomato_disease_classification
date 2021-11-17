# eff-tomato_disease_classification
 - Writer : Kiwon Seo

<br>
<hr>
<br>

## 1. 개요
 - 데이터 : 토마토 잎사귀 데이터
 - 과제 : 토마토 병충해 종류 분류 (10종류)
 - train/images : 13,861장 이미지
 - test/images : 3463장 이미지
 - Efficientnet : 모델의 정확도와 밀접한 관계의 깊이, 너비, 입력 이미지 크기 간의 최적의 관계를 정의한 모델

<br>
<hr>
<br>

## 2. How to use
 - Run : python train.py
 - Predict : python predict.py
 - Ensemble : python ensemble.py
 
<br>
<hr>
<br>

## 3. 확장 - WanDB
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
 - ID : 깃헙 계정
 - Full name : Ben Seo   
 
    Username : benseo
    
    Organization : Kyowon
    
    Project Name : eff-tomato_disease_classification
    
    Project : https://wandb.ai/benseo/eff-tomato_disease_classification
    
    Project Run : https://wandb.ai/benseo/eff-tomato_disease_classification/runs/실험-이름

<br>

### (3) WanDB 코드 튜토리얼
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

## 4. 성능 개선 및 실험
### (1) 모델 버전 실험
 - Pretrained Model : EfficientNet b0, b3, b6
 - 모델 별로 구조, 채널, 깊이 넓이가 다르기 때문에 모델 버전을 바꿔 가며 가벼운 모델 부터 무거운 모델 까지 실
 - 레이어 : 1280 -> 500 -> 250 -> 10
 - 결과 : 성능 향상
```python
# - EfficientNet은 논문에서 모델 버전 별로 최적의 Input Shape을 제시 함 (버전 별로 Input Shape 지정 필요)
# - 모델을 freeze 시켜서 탑 레이어를 쌓아 실험 이후 unfreeze 하여 학습 하는 방법 시도
# - 레이어 실험 : 1280 - 500 - 250 - 10
class PestClassifier(nn.Module):
    def __init__(self, num_class):
        super(PestClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1280)

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
        
    def forward(self, input_img):
        # Model
        x = self.model(input_img)
        
        return x
```

<br>

### (2) Image Augmentation 실험
 - Image Augmentation은 원본 이미지를 조작 하여 원본에 크고 작은 변화들을 주어 대상 고유의 특징을 조금 더 넓게 가지게 할 수 있는 기법으로 Image 문제에서 Overfitting을 방지 하고 성능을 향상 할 수 있는 기법
 - 실험 : 이미지 랜덤 회전, 랜덤 수평 뒤집기, 랜덤 수직 뒤집기 등
 - 결과 : 성능 향상
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

### (3) Early Stopping 실험
 - Early Stopping : 과적합을 회피 하도록 만든 기법으로, Epoch마다 Validation Loss를 체크 해서 로스가 떨어지지 않는 구간을 count 하여 일정 patience에 다다르면 훈련을 조기에 멈추는 방식을 이용 (과적합이 발생 하기 전 까지는 training loss와 validation loss가 둘 다 감소 하지만, 과적합이 일어나면 training loss는 감소하는 반면에 validation loss는 증가)
 - 실험 목적 : 진동이 많을 경우 성급한 조기 종료가 일어 날 수 있으므로 Patience를 조절 하며 실험 진행
 - 결과 : 성능 향상
```bash
TRAIN:
  num_epochs: 50
  batch_size: 128
  learning_rate: 0.0005
  early_stopping_patience: 50 # early stopping patience
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

### (4) Softmax
 - 클래스 분류 문제에서 마지막 단계에 출력 값을 정규화 해 주는 함수로, 마지막 단에서 출력을 0~1 사이 값으로 정규화 하여 총 합을 1이 되게 해 주는 함수
 - PyTorch의 CrossEntropy Loss는 softmax와 CrossEntropy Loss를 합친 것을 제공 하므로 Softmax를 떼고 실험
 - 결과 : 성능 향상

<br>

### (5) Initialization
 - Initialization : 초기 값을 어떻게 선택 하느냐에 따라 학습에서 다른 성능을 보임
 - 실험 : Xavier, Kaiming 등 실험
 - 결과 : 성능 향상 X

<br>

### (6) Input Size를 모델 버전 별 권장 사이즈로 적용 하기
 - 결과 : 성능 향상

<br>

### (7) Ensemble
 - Weighted Voting Ensemble 진행 (모델 버전 별, 성능 높은 모델 별)
 - 추후 아웃풋 기반의 앙상블이 아닌, 모델 별 output을 확률 값으로 출력 해서 mean으로 Majority Voting Ensemble 적용 하기.

<br>
<hr>
<br>

## 5. 기타
 - Colab GPU or TPU로 학습 하기 (TPU로 이용 할 경우 데이터셋 변환이 필요 하나 속도 뿐 아니라 계산 성능도 좋아짐. float32 -> float64를 계산 할 경우 소숫점 밑 64자리를 더 정확히 계산하게 되기 때문)
 - Test set에도 Image Augmentation 적용 하는 TTA 실험 하기
