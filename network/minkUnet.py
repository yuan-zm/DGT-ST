
from turtle import forward
from network.resnet import ResNetBase
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
import os
import MinkowskiEngine.MinkowskiFunctional as MEF
# from .network_utils import *

from MinkowskiSparseTensor import SparseTensor

import torch.nn.functional as F

def abs_me(inputs):
    return SparseTensor(
        torch.abs(inputs.F),
        coordinate_map_key=inputs.coordinate_map_key,
        coordinate_manager=inputs.coordinate_manager,
    )


def mean_me(inputs, dim=1, keepdim=True):
    return SparseTensor(
        torch.mean(inputs.F, dim=dim, keepdim=keepdim),
        coordinate_map_key=inputs.coordinate_map_key,
        coordinate_manager=inputs.coordinate_manager,
    )


class depth_reg(nn.Module):
    def __init__(self, D=3):
        super().__init__()

        # ---------------------Start Decoder for intensity restruction channel is 128---------------------
        # layer 2
        self.reg_convtr6p4s2 = ME.MinkowskiConvolutionTranspose(128, 96,
                                                                kernel_size=2,
                                                                stride=2, dimension=D)
        self.reg_bntr6 = ME.MinkowskiBatchNorm(96)
        
        self.enc_1_remi = nn.Sequential(ME.MinkowskiConvolution(96, 64,  # 128  -> 64
                                                                kernel_size=3,
                                                                stride=1, dimension=D,
                                                                bias=False),
                                        ME.MinkowskiBatchNorm(64)
                                        )

        #  layer 1
        self.reg_convtr7p2s2 = ME.MinkowskiConvolutionTranspose(64, 64,
                                                                kernel_size=2, stride=2,
                                                                dimension=D)
        self.reg_bntr7 = ME.MinkowskiBatchNorm(64)
       
        self.enc_2_remi = nn.Sequential(ME.MinkowskiConvolution(64, 32,  # 128  -> 64
                                                                kernel_size=3,
                                                                stride=1, dimension=D,
                                                                bias=False),
                                        ME.MinkowskiBatchNorm(32)
                                        )
        # 返回这个特征到main decoder
        self.dec_remi = nn.Sequential(ME.MinkowskiConvolution(32, 96,
                                                              kernel_size=1, stride=1, dimension=D,
                                                              bias=False),
                                      ME.MinkowskiBatchNorm(96)
                                      )
        # --------------------- End Decoder for intensity restruction ---------------------
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out_remi = self.reg_convtr6p4s2(x)
        out_remi = self.reg_bntr6(out_remi)
        out_remi = self.relu(out_remi)

        out_remi = self.enc_1_remi(out_remi)
        out_remi = self.relu(out_remi)

        # ----------- 

        out_remi = self.reg_convtr7p2s2(out_remi)
        out_remi = self.reg_bntr7(out_remi)
        out_remi = self.relu(out_remi)
        
        out_remi = self.enc_2_remi(out_remi)
        out_remi = self.relu(out_remi)

        intentsity_res = mean_me(out_remi)

        out_remi = self.dec_remi(out_remi)
        remi2seg = self.relu(out_remi)

        return intentsity_res, remi2seg




class att_head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ---------------------Start Decoder for intensity restruction channel is 128---------------------
        # layer 2
        self.att_conv = nn.Linear(96, 1)
        # self.sigmoid = ME.MinkowskiSigmoid()
        # # --------------------- End Decoder for intensity restruction ---------------------
        # self.relu = ME.MinkowskiReLU(inplace=True)

        self.cls = nn.Linear(96, num_classes)

    def forward(self, x, x_4, domain='src'):
        if domain == 'src':
            out = self.cls(x)
        else:
            att_weight = F.sigmoid(self.att_conv(x))
            fu_ft = att_weight * x + (1 - att_weight) * x_4
            out = self.cls(fu_ft)

        return out

class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, out_level=False, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, out_level, D)

    def network_initialization(self, in_channels, out_channels, out_level, D):
        self.out_level = out_level
        # Output of the first conv concated to conv6
        # Source domain specific encoder
        # Source domain specific encoder
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])
       
        # domain shared
        self.conv3p4s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[4], kernel_size=2, stride=2,
                                                             dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[5], kernel_size=2, stride=2,
                                                            dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])

        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])


        self.final = ME.MinkowskiConvolution(self.PLANES[7] * self.BLOCK.expansion,
                                            out_channels, kernel_size=1,
                                            bias=True, dimension=D)

        self.relu = ME.MinkowskiReLU(inplace=True)

        self.interp = ME.MinkowskiInterpolation()
        
    def forward(self, x, is_train=True):
        # ########################Sparse Conv #################
      
        out = self.conv0p1s1(x) # 4 -> 32
        out = self.bn0(out)
        out_p1 = self.relu(out)
        # tensor_stride=2
        out = self.conv1p1s2(out_p1) # 32 - > 32 torch.Size([80649, 32])
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        # tensor_stride=4
        out = self.conv2p2s2(out_b1p2) # 32 - > 32 torch.Size([38127, 32])
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out) # 32 - > 64

        # tensor_stride=8
        out = self.conv3p4s2(out_b2p4) # 64 - > 64
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out) # 64 - > 128

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)  # 128 - > 128
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out) # 128 - > 256

        # tensor_stride=8
        out = self.convtr4p16s2(out)  # 256 - > 128
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out) # 256 - > 128
        
        # tensor_stride=4
        out = self.convtr5p8s2(out) # 128 - > 128
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out_4 = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out_4)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        # out = out + remi2seg
        out = ME.cat(out, out_p1)
        y4 = self.block8(out)
      
        sp_out = self.final(y4)

        

        if self.out_level and is_train:
            out_feat = y4.F
            return sp_out, out_feat
        else:
            return sp_out


class MinkUNet14A(MinkUNetBase):
    BLOCK = BasicBlock
    # LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    # PLANES = (32, 64, 128, 256, 128, 128, 96, 96)
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)

class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

if __name__ == '__main__':
    from tests.python.common import data_loader

    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = MinkUNet18A(in_channels=3, out_channels=5, D=3)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader(is_classification=False)
        input = ME.SparseTensor(feat, coordinates=coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))
