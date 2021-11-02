"""
"""
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

########################################### 기본 모델 ###########################################
# class PestClassifier(nn.Module):
#     def __init__(self, num_class):
#         super(PestClassifier, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_class) # b0 -> b6
#         #self.softmax = nn.Softmax(dim=1)
                     
#     def forward(self, input_img):
#         x = self.model(input_img)
#         #x = self.softmax(x)
#         return x
#################################################################################################

########################################### 레이어 쌓은 모델 ###########################################
#efficientnet 논문 : eff 모델 버전 별로 input shape에 대한 실험적인 결과를 다룬 논문 -> 버전 별로 input shape 설정 필요
#efficientnet은 일반적으로 모델을 freeze 시켜서 top layer를 쌓아 실험 하고 이후 unfreeze 시킨 후 학습 하는 방법이 이용 됨
#b3 try (reference : https://www.kaggle.com/akasharidas/plant-pathology-2020-in-pytorch)
# reference 2 : https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
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
#################################################################################################

########################################### 레이어 쌓은 모델 + 케이밍 ###########################################
# # 참고 1 : https://www.kaggle.com/akasharidas/plant-pathology-2020-in-pytorch
# # 참고 2 : https://github.com/khornlund/aptos2019-blindness-detection/blob/35192b07c42aa2e06aac531ac017a5bc4ce54f00/aptos/model/model.py
# class PestClassifier(nn.Module):
#     def __init__(self, num_class):
#         super(PestClassifier, self).__init__()
#         self.model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=1000)

#         self.model._fc = nn.Sequential(OrderedDict([
#             ('linear1', nn.Linear(self.model._fc.in_features, 512)),
#             ('relu1', nn.ReLU()),
#             ('drop1', nn.Dropout(p=0.5)),
#             ('linear2', nn.Linear(512, 256)),
#             ('relu2', nn.ReLU()),
#             ('drop2', nn.Dropout(p=0.25)),
#             ('linear3', nn.Linear(256, num_class))
#         ]))

#         nn.init.kaiming_normal_(self.model._fc._modules['linear1'].weight)
#         nn.init.kaiming_normal_(self.model._fc._modules['linear2'].weight)
#         nn.init.kaiming_normal_(self.model._fc._modules['linear3'].weight)
                
#     def forward(self, input_img):
#         # Model
#         x = self.model(input_img)
        
#         return x
#################################################################################################