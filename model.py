import torch
import torch.nn as nn
import torch.nn.functional as F


#test
from torch.autograd.variable import Variable

"""
The architecture of f consists of four convolutional layers with 64, 128, 256, 128 filters respectively,
each followed by max pooling and ReLU non-linearity.
"""
class _f(nn.Module):

    #input (batch_size, 3, 32, 32)
    #ouput (batch_size, 128, 1, 1)
    #ouput_pretrain (batch_size, 10)
    def __init__(self, mode):
        super(_f, self).__init__()
        self.mode = mode

        #model
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.final_pool = nn.MaxPool2d(4)
        self.conv5 = nn.Conv2d(128, 10, 1, 1)

    def forward(self, input):
        if(input.shape[1] == 1):
            input = torch.cat((input, input, input), 1)
        x = self.pool(F.relu(self.conv1(input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.final_pool(self.conv4(x)))

        if self.mode == 'pretrain':
            x = self.conv5(x)
        return x

"""
Network g, inspired by Radford et al. (2015), maps SVHN-trained f’s 128D representations to 32×
32 grayscale images. g employs four blocks of deconvolution, batch-normalization, and ReLU, with
a hyperbolic tangent terminal. 
"""
class _g(nn.Module):

    #input (batch_size, 128, 1, 1)
    def __init__(self):
        super(_g, self).__init__()
        #input_channel, output_channel, kernel_size, stride, padding
        self.deconv1 = nn.ConvTranspose2d(128, 256, 4, 1, 0)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.batch_norm4 = nn.BatchNorm2d(1)

    def forward(self, input):
        x = F.relu(self.batch_norm1(self.deconv1(input)))
        x = F.relu(self.batch_norm2(self.deconv2(x)))
        x = F.relu(self.batch_norm3(self.deconv3(x)))
        x = F.tanh(self.batch_norm4(self.deconv4(x)))
        return x

"""
The architecture of D consists of four batch-normalized convolutional
layers and employs ReLU
"""

class _D(nn.Module):

    #input (batch_size, 1, 32, 32)
    #output (batch_size 3, 1, 1)
    def __init__(self):
        super(_D, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 2, 1)
        self.batchNorm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.batchNorm3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 3, 4, 4, 1)
        self.batchNorm4 = nn.BatchNorm2d(3)

    def forward(self, input):
        x = F.relu(self.batchNorm1(self.conv1(input)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = x.view(-1, 3)
        return x