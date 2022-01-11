# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2021/12/29
"""
FLGCNN: A novel fully convolutional neural network for end-to-end monaural speech enhancement with utterance-based objective functions
不知道论文中的CSTFT/ CSTFT是不是 DCCRN中的ConvSTFT、ConviSTFT，暂且借过来用吧,也可以使用torch.stft\istft
"""
import torch

from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            return real, imag


class ConviSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', fix=True):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs):
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]

        return outputs


class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation (default activation:nn.LeakyReLU(0.2, inplace=True))
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=nn.PReLU(num_parameters=1, init=0.2)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                     bias)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class GatedDeConv2dWithActivation(nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True,
                 batch_norm=True, activation=nn.PReLU(num_parameters=1, init=0.2)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        # (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.Dconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                          groups, bias, dilation)
        self.mask_Dconv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                               groups, bias, dilation)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.Dconv2d(input)
        mask = self.mask_Dconv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels, bias=False)

        pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        if causal:
            self.net = nn.Sequential(depthwise_conv,
                                     Chomp1d(padding),
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv,
                                     nn.PReLU(),
                                     nn.BatchNorm1d(in_channels),
                                     pointwise_conv)

    def forward(self, x):
        return self.net(x)


class ResTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResTemporalBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(num_features=out_channels),
            DepthwiseSeparableConv(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=1,
                                   padding=(kernel_size - 1) * dilation, dilation=dilation, causal=True)
        )

    def forward(self, input):
        x = self.block(input)
        return x + input


# TemporalConvNet
# 输入：(B, C_in, L_in)
# 输出：(B, C_in, L_out)
class ResTemporal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 init_dilation=3, num_layers=6):
        super(ResTemporal, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation ** i
            # in_channels = in_channels if i == 0 else out_channels

            layers += [ResTemporalBlock(in_channels, out_channels, kernel_size,
                                        dilation=dilation_size)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# input:(B,1,T,F)
# output:(B,1,T,F)
class FLGCNN(nn.Module):
    def __init__(self):
        super(FLGCNN, self).__init__()
        self.CSTFT = ConvSTFT(win_len=512, win_inc=256, fft_len=512, win_type='hanning', feature_type='real')
        self.GConv2d_1 = GatedConv2dWithActivation(in_channels=2, out_channels=16, kernel_size=(3, 5), stride=(1, 1),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.GConv2d_2 = GatedConv2dWithActivation(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 1), batch_norm=False, activation=None)
        self.GConv2d_3 = GatedConv2dWithActivation(in_channels=16, out_channels=16, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.GConv2d_4 = GatedConv2dWithActivation(in_channels=16, out_channels=32, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.GConv2d_5 = GatedConv2dWithActivation(in_channels=32, out_channels=32, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.GConv2d_6 = GatedConv2dWithActivation(in_channels=32, out_channels=64, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.GConv2d_7 = GatedConv2dWithActivation(in_channels=64, out_channels=64, kernel_size=(3, 5), stride=(1, 2),
                                                   padding=(1, 2), batch_norm=False, activation=None)
        self.TCM_1 = ResTemporal(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCM_2 = ResTemporal(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.TCM_3 = ResTemporal(in_channels=256, out_channels=512, kernel_size=3, init_dilation=2, num_layers=6)
        self.GDConv2d_7 = GatedDeConv2dWithActivation(in_channels=256, out_channels=64, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 2), output_padding=(0, 1),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_6 = GatedDeConv2dWithActivation(in_channels=256, out_channels=32, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 2), output_padding=(0, 1),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_5 = GatedDeConv2dWithActivation(in_channels=128, out_channels=32, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 2), output_padding=(0, 1),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_4 = GatedDeConv2dWithActivation(in_channels=128, out_channels=16, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 2), output_padding=(0, 1),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_3 = GatedDeConv2dWithActivation(in_channels=64, out_channels=16, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 2), output_padding=(0, 1),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_2 = GatedDeConv2dWithActivation(in_channels=64, out_channels=16, kernel_size=(3, 5),
                                                      stride=(1, 2), padding=(1, 1), output_padding=(0, 0),
                                                      batch_norm=False, activation=None)
        self.GDConv2d_1 = GatedDeConv2dWithActivation(in_channels=64, out_channels=2, kernel_size=(3, 5),
                                                      stride=(1, 1), padding=(1, 2), output_padding=(0, 0),
                                                      batch_norm=False, activation=None)  # 论文中 stride(1,2)我怀疑写错了
        self.CISTFT = ConviSTFT(win_len=512, win_inc=256, fft_len=512, win_type='hanning')

    def skip_connect(self, encode, decode):
        one = torch.cat((encode, decode), dim=1)

        encode = torch.sigmoid(encode)
        two = torch.cat((encode, decode), dim=1)

        output = torch.cat((one, two), dim=1)
        return output

    def forward(self, x):
        real, imag = self.CSTFT(x)
        x = torch.stack((real, imag), dim=1).transpose(-1,-2)
        # print(x.shape)

        GConv2d_1 = self.GConv2d_1(x)
        # print("GConv2d_1", GConv2d_1.shape)  # [64, 16, 32, 257]
        GConv2d_2 = self.GConv2d_2(GConv2d_1)
        # print("GConv2d_2", GConv2d_2.shape)  # [64, 16, 32, 128]
        GConv2d_3 = self.GConv2d_3(GConv2d_2)
        # print("GConv2d_3", GConv2d_3.shape)  # [64, 16, 32, 64]
        GConv2d_4 = self.GConv2d_4(GConv2d_3)
        # print("GConv2d_4", GConv2d_4.shape)  # [64, 32, 32, 32]
        GConv2d_5 = self.GConv2d_5(GConv2d_4)
        # print("GConv2d_5", GConv2d_5.shape)  # [64, 32, 32, 16]
        GConv2d_6 = self.GConv2d_6(GConv2d_5)
        # print("GConv2d_6", GConv2d_6.shape)  # [64, 64, 32, 8]
        GConv2d_7 = self.GConv2d_7(GConv2d_6)
        # print("GConv2d_7", GConv2d_7.shape)  # [B, 64, 32, 4] [batch_size,C,T,F]
        reshape_2 = GConv2d_7.permute(0, 1, 3, 2)
        batch_size, C, F, T = reshape_2.shape
        reshape_2 = reshape_2.reshape(batch_size, C * F, T)
        # print("reshape_2", reshape_2.shape)  # [64, 256, T]
        TCM_1 = self.TCM_1(reshape_2)
        TCM_2 = self.TCM_2(TCM_1)
        TCM_3 = self.TCM_3(TCM_2)
        reshape_3 = TCM_3.reshape(batch_size, C, F, T)
        reshape_3 = reshape_3.permute(0, 1, 3, 2)  # (B,C,T,F)
        # print("reshape_3", reshape_3.shape)  # [64, 64, 32, 4]
        skip_1 = self.skip_connect(GConv2d_7, reshape_3)
        # print("skip_1", skip_1.shape)  # [64, 256, 32, 4]
        GDConv2d_7 = self.GDConv2d_7(skip_1)
        # print("GDConv2d_7", GDConv2d_7.shape)  # [64, 64, 32, 8]

        skip_2 = self.skip_connect(GConv2d_6, GDConv2d_7)
        # print("skip_2", skip_2.shape)       # [64, 256, 32, 8]
        GDConv2d_6 = self.GDConv2d_6(skip_2)
        # print("GDConv2d_6", GDConv2d_6.shape)   # [64, 32, 32, 16]

        skip_3 = self.skip_connect(GConv2d_5, GDConv2d_6)
        # print("skip_3", skip_3.shape)       # [64, 128, 32, 16]
        GDConv2d_5 = self.GDConv2d_5(skip_3)
        # print("GDConv2d_5", GDConv2d_5.shape)   # [64, 32, 32, 32]

        skip_4 = self.skip_connect(GConv2d_4, GDConv2d_5)
        # print("skip_4", skip_4.shape)       # [64, 128, 32, 32]
        GDConv2d_4 = self.GDConv2d_4(skip_4)
        # print("GDConv2d_4", GDConv2d_4.shape)   # [64, 16, 32, 64]

        skip_5 = self.skip_connect(GConv2d_3, GDConv2d_4)
        # print("skip_5", skip_5.shape)       # [64, 64, 32, 64]
        GDConv2d_3 = self.GDConv2d_3(skip_5)
        # print("GDConv2d_3", GDConv2d_3.shape)       # [64, 16, 32, 128]

        skip_6 = self.skip_connect(GConv2d_2, GDConv2d_3)
        # print("skip_6", skip_6.shape)       # [64, 64, 32, 128]
        GDConv2d_2 = self.GDConv2d_2(skip_6)
        # print("GDConv2d_2", GDConv2d_2.shape)   # [64, 16, 32, 257]

        skip_7 = self.skip_connect(GConv2d_1, GDConv2d_2)
        # print("skip_7", skip_7.shape)       # [64, 64, 32, 257]
        GDConv2d_1 = self.GDConv2d_1(skip_7)
        print("GDConv2d_1", GDConv2d_1.shape)   # [64, 1, 32, 257]

        GDConv2d_1 = GDConv2d_1.transpose(-1, -2)
        output = self.CISTFT(torch.cat((GDConv2d_1[:,0,:,:],GDConv2d_1[:,1,:,:]),dim=1))

        return output


if __name__ == "__main__":
    # x = torch.rand(64, 1, 8192)
    # model = ResTemporalBlock(in_channels=1, out_channels=256, kernel_size=3, padding=6, dilation=3)
    # model = ResTemporal(in_channels=64,out_channels=64,kernel_size=3,init_dilation=3,num_layers=3)

    x = torch.rand(64, 8192)  # (B,C,T,F)
    # model = GatedConv2dWithActivation(in_channels=1, out_channels=16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2),
    #                                   batch_norm=False, activation=None)
    model = FLGCNN()
    # summary(model, input_size=(64, 1, 32, 257))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    output = model(x)
    print("output", output.shape)
    # torch.save(model.state_dict(),"FLGCNN.pth") # 20M
