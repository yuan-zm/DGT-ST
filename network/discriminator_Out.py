import torch.nn as nn
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

import numpy as np

from MinkowskiSparseTensor import SparseTensor

import torch.nn as nn
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

import numpy as np

def log2_me(inputs):
    """
    这个函数是用来对Minko库的补充
    return torch.log2(inputs.F + 1e-30)
    """
    return SparseTensor(
        torch.log2(inputs.F + 1e-30),
        coordinate_map_key=inputs.coordinate_map_key,
        coordinate_manager=inputs.coordinate_manager,
    )

def div_me(inputs, div_num):
    """
    这个函数是用来对Minko库的补充
    return inputs.F / div_num
    """
    return SparseTensor(
        inputs.F / div_num,
        coordinate_map_key=inputs.coordinate_map_key,
        coordinate_manager=inputs.coordinate_manager,
    )

class Discriminator_out(nn.Module):

    def __init__(self, cfg):
        super(Discriminator_out, self).__init__()
        self.cfg = cfg
        # 总共5次卷积  5次下采样
        self.layers = cfg.MODEL_D.N_LAYERS - 1

        strides = cfg.MODEL_D.DOWN_SAMPLE_TIMES
        self.d_out = cfg.MODEL_D.FEATURE_CHANNELS
        d_in = cfg.MODEL_D.IN_CHANNELS
        k_size = cfg.MODEL_D.DIS_KERNEL_SIZSE

        # ---------- Define Sparse Conv ---------- #
        self.model_blocks = nn.ModuleList()
        for i in range(self.layers):  # 40960 10280 2570 640
            d_out = self.d_out[i]
            self.model_blocks.append(self.get_conv_block(
                d_in, d_out, stride=strides[i], kernel_size=k_size))
            d_in = d_out
        
        d_out = self.d_out[-1]
        self.model_blocks.append(
            ME.MinkowskiConvolution(d_in, d_out,
                                    kernel_size=k_size, stride=strides[-1],
                                    dimension=3)
            )

        self.interp = ME.MinkowskiInterpolation()
        # # ----------End Define Sparse Conv ---------- #
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
                
    def get_conv_block(self, in_channel, out_channel, kernel_size, stride=1):
        return nn.Sequential(ME.MinkowskiConvolution(in_channel, out_channel,
                                                     kernel_size=kernel_size, stride=stride,
                                                     dimension=3),
                             ME.MinkowskiLeakyReLU(),
                             )

    def forward(self, x):

        # ----------Start Sparse Conv ---------- #
        n, c = x.size()

        features = MEF.softmax(x)
        if self.cfg.MODEL_D.IS_ADVENT:
            features = div_me((features * log2_me(features)), np.log2(c))

        for i in range(self.layers):
            features = self.model_blocks[i](features)
        
        features = self.model_blocks[-1](features)

        #  Transpose the bottem feature map to origin size
        end_out = self.interp(features, x.C.float())

        return end_out



