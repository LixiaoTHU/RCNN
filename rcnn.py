#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rcnnblock import RCL

class RCNN(nn.Module):
    def __init__(self, channels, num_classes, K = 96, steps = 4):
        super(RCNN, self).__init__()
        self.K = K

        self.layer1 = nn.Conv2d(channels, K, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(K)
        self.pooling1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.pooling2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer2 = RCL(K, steps=steps)
        self.layer3 = RCL(K, steps=steps)
        self.layer4 = RCL(K, steps=steps)
        self.layer5 = RCL(K, steps=steps)
        
        self.fc = nn.Linear(K, num_classes, bias = True)
        self.dropout = nn.Dropout(p=0.5)

        #init the parameter    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(self.relu(x))
        x = self.pooling1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    import numpy as np

    transform = transforms.Compose([
        transforms.TenCrop(24),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))])

    trainset = CIFAR10("~/Datasets/", transform = transform, download=True)
    testloader = torch.utils.data.DataLoader(trainset, batch_size = 32, shuffle = False, num_workers = 32)
    net = RCNN(3, 10, K = 96)
    for (images, labels) in testloader:
        bs, ncrops, c, h, w = images.size()
        result = net(images.view(-1, c, h, w))
        result_avg = result.view(bs, ncrops, -1).mean(1)
        print(result_avg.shape)
        print(labels.shape)
        break

    size = 1
    for param in net.parameters():
        arr = np.array(param.size())

        s = 1
        for e in arr:
            s *= e

        size += s

    print("all parameters %.2fM" %(size/1e6) )