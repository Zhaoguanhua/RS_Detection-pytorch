# Copyright (c) Facebook, Inc. and its affiliates.
# import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init




class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class CoarseMaskHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, num_classes,input_channel=256,input_height=14,input_width=14):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(CoarseMaskHead, self).__init__()

        # fmt: off
        self.num_classes            = num_classes
        conv_dim                    = 256
        self.fc_dim                 = 1024
        num_fc                      = 2
        self.output_side_resolution = 7
        self.input_channels         = input_channel
        self.input_h                = input_height
        self.input_w                = input_width
        # fmt: on

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                activation=F.relu,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution

        self.prediction = nn.Linear(self.fc_dim, output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        # unlike BaseMaskRCNNHead, this head only outputs intermediate
        # features, because the features will be used later by PointHead.
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N, self.num_classes, self.output_side_resolution, self.output_side_resolution
        )
