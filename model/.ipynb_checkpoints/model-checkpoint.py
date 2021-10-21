"""
"""
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet

# # b6 try
# class PestClassifier(nn.Module):
#     def __init__(self, num_class):
#         super(PestClassifier, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_class) # b0 -> b6
#         self.softmax = nn.Softmax(dim=1)
                     
#     def forward(self, input_img):
#         x = self.model(input_img)
#         x = self.softmax(x)
#         return x

#efficientnet 논문 : eff 모델 버전 별로 input shape에 대한 실험적인 결과를 다룬 논문 -> 버전 별로 input shape 설정 필요
#efficientnet은 일반적으로 모델을 freeze 시켜서 top layer를 쌓아 실험 하고 이후 unfreeze 시킨 후 학습 하는 방법이 이용 됨
#b3 try (reference : https://www.kaggle.com/akasharidas/plant-pathology-2020-in-pytorch)
class PestClassifier(nn.Module):
    def __init__(self, num_class):
        super(PestClassifier, self).__init__()
        # Pretrained Model : efficientnet-b3
        self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1000)

        # Top Layer : effificientnet-b3에 fully-connected 레이어를 쌓아서 최종 레이어에서는 10개가 output (클래스가 10개임)
        # Pretrained model 이후 쌓는 레이어들로, 실험적 혹은 경험적 결과로 레이어를 쌓으면 됨
        num_features = self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Linear(num_features, 512),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.25),
                                 nn.Linear(256, num_class))

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