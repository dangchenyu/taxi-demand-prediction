from regressor.res18 import resnet18
import torch.nn.functional as F
import numpy
import torch
import torch.nn as nn
from torchvision import models
import cv2
from torchvision import transforms

class Box_reg(nn.Module):
    def __init__(self,):
        super(Box_reg, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.regresssion = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    def forward(self, input):
        input=self.backbone(input)
        input=input.view(input.size(0),-1)
        output=self.regresssion(input)
        output=torch.sigmoid(output)
        return output



if __name__ == '__main__':
    model=Box_reg()
    dumy=torch.rand(1,3,64,64)
    output=model(dumy)