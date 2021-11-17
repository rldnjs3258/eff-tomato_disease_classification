"""
"""
import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict

########################################### 1. 기본 모델 ###########################################
# - 모델 : EfficientNet b6
# - Softmax : N (Pytorch의 Crossentropy 로스에는 Softmax가 붙어 있음)
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

########################################### 2. 탑 레이어 쌓은 모델 ###########################################
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
#################################################################################################

########################################### 3. 탑 레이어 쌓은 모델 + 케이밍 ###########################################
# - Initialization을 케이밍으로 바꿔서 실행 진행
# - 레이어 실험 : 1000 - 512 -256 - 10
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