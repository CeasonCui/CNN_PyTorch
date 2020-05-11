import cv2
from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from PIL import Image
import matplotlib.pyplot as plt


EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 32
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
channel = 8

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=channel,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (32, 64, 64)
            nn.ReLU(),                      # activation
            #nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 32, 32)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 32, 32)
            nn.Conv2d(channel, channel*2, 3, 1, 1),     # output shape (64, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (64, 16, 16)
        )
        self.conv3 = nn.Sequential(         # input shape (64, 16, 16)
            nn.Conv2d(channel*2, channel*4, 3, 1, 1),     # output shape (128, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (128, 8, 8)
        )
        self.conv4 = nn.Sequential(         # input shape (128, 8, 8)
            nn.Conv2d(channel*4, channel*8, 3, 1, 1),     # output shape (256, 8, 8)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (256, 4, 4)
        )
        self.fc1 = nn.Linear(channel*4 * 8 * 8, 2)   # fully connected layer, output 2 classes
        self.pool = nn.MaxPool2d(2, stride=2)
    def forward(self, x):
        x = x.float()
        x = x.view(-1, 1, 64, 64)
        #x = x.reshape(-1, 1, 64, 64)
        x = self.conv1(x)
        x1 = x.reshape(-1, 1, 64, 64)
        x = self.pool(x)
        #x1 = x1.cpu().numpy()
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.fc1(x)
        #output = self.softmax(x)
        return  x1   # return x for visualization

cnn2 = CNN()
cnn2.load_state_dict(torch.load('cnn8_3.pth'))
path = './dataset_6/dataset/ellipse_d2_p1378.jpg'
img = io.imread(path)
train_loader = DataLoader(img,batch_size=64,shuffle=False)#使用DataLoader加载数据
for i_batch,batch_data in enumerate(train_loader):
    print(batch_data.size())
    feature = cnn2(batch_data)
    print(feature.size())
    print(feature[1].size())
    for i in range(8):
        image = feature[i].cpu().clone()
        image = image.squeeze(0)
        image1 = transforms.ToPILImage()(image)
        image1.save('feature_'+str(i+1)+'.jpg', quality=95)
        #feature = feature[i].detach().numpy()
        #cv2.imwrite('./feature.jpg',feature)
# feature = feature.numpy()
# cv2.imwrite('./feature.jpg',feature)
