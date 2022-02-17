import functools

from torch import nn
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import json
from skimage.transform import resize

import pdb

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Interfaces
##################################################################################

def get_generator(gen_opt, train_mode):
    """Get a generator, gen_opt is options of gen, train mode determine which generator model to use"""
    if 'direct_intrinsics' in train_mode:
        return DirectIntrinsics(gen_opt)
    else:
        raise NotImplementedError


def get_manual_criterion(name):
    if 'distance' in name.lower():
        return None
    else:
        raise NotImplementedError(name + 'should in [distance/...]')


##################################################################################
# Generator
##################################################################################
class DirectIntrinsics(nn.Module):
    """ repeat net structure in DI paper """
    def __init__(self, gen_opt):
        super(DirectIntrinsics, self).__init__()
        self.norm = gen_opt.norm
        self.pad_type = gen_opt.pad_type
        self.input_dim = gen_opt.input_dim
        self.output_dim_s = gen_opt.output_dim_s  # 1
        self.output_dim_r = gen_opt.output_dim_r  # 3

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        """scale 1 sub net"""
        model_s1 = []  # scale 1
        # conv_1
        model_s1 += [Conv2dBlock(self.input_dim, 96, kernel_size=11, stride=4, padding=5,
                                 pad_type=self.pad_type, activation='relu', norm=self.norm, use_drop=False)]
        model_s1 += [self.maxpool]  # 1/2 * 1/4 = 1/8 size
        # conv_2
        model_s1 += [Conv2dBlock(96, 256, kernel_size=5, stride=1, padding=2,
                                 pad_type=self.pad_type, activation='relu', norm=self.norm, use_drop=False)]
        model_s1 += [self.maxpool]  # 1/16 size
        # conv_3
        model_s1 += [Conv2dBlock(256, 384, kernel_size=3, stride=1, padding=1,
                                 pad_type=self.pad_type, activation='relu', use_drop=False)]
        # conv_4
        model_s1 += [Conv2dBlock(384, 384, kernel_size=3, stride=1, padding=1,
                                 pad_type=self.pad_type, activation='relu', use_drop=False)]
        # conv_5
        model_s1 += [Conv2dBlock(384, 256, kernel_size=3, stride=1, padding=1,
                                 pad_type=self.pad_type, activation='relu', use_drop=False)]
        model_s1 += [self.maxpool]  # 1/32 size

        # upsample_5
        model_s1 += [nn.ConvTranspose2d(256, 256, kernel_size=16, stride=8, padding=4)]  # 1/4 size

        # out_scale1
        model_s1 += [Conv2dBlock(256, 64, kernel_size=1, stride=1, padding=0,
                                 pad_type=self.pad_type, activation='prelu', use_drop=True)]
        self.model_s1 = nn.Sequential(*model_s1)

        """scale 2 sub net"""
        model_s2 = []  # scale 2
        # conv_2-1
        model_s2 += [Conv2dBlock(3, 96, kernel_size=9, stride=2, padding=4,
                                 pad_type=self.pad_type, activation='prelu', use_drop=False)]
        model_s2 += [self.maxpool]  # 1/2 * 1/2 = 1/4 size
        self.model_s2 = nn.Sequential(*model_s2)

        """combine s1 and s2 after concat"""
        model_combine = []
        # conv_2-2
        model_combine += [Conv2dBlock(160, 64, kernel_size=5, stride=1, padding=2,
                                      pad_type=self.pad_type, activation='prelu', use_drop=True)]
        # conv_2-3
        model_combine += [Conv2dBlock(64, 64, kernel_size=5, stride=1, padding=2,
                                      pad_type=self.pad_type, activation='prelu', use_drop=True)]
        # conv_2-4
        model_combine += [Conv2dBlock(64, 64, kernel_size=5, stride=1, padding=2,
                                      pad_type=self.pad_type, activation='prelu', use_drop=True)]
        self.model_combine = nn.Sequential(*model_combine)

        """output albedo sub net"""
        model_albedo = []
        # conv_2-5_a
        model_albedo += [Conv2dBlock(64, 64, kernel_size=5, stride=1, padding=2,
                                      pad_type=self.pad_type, activation='prelu', use_drop=True)]
        model_albedo += [nn.ConvTranspose2d(64, self.output_dim_r, kernel_size=8, stride=4, padding=2)]  # original size
        self.model_albedo = nn.Sequential(*model_albedo)

        """output shading sub net"""
        model_shading = []
        # conv_2-5_s
        model_shading += [Conv2dBlock(64, 64, kernel_size=5, stride=1, padding=2,
                                     pad_type=self.pad_type, activation='prelu', use_drop=True)]
        model_shading += [nn.ConvTranspose2d(64, self.output_dim_s, kernel_size=8, stride=4, padding=2)]  # original size
        self.model_shading = nn.Sequential(*model_shading)

    def forward(self, x):
        x_s1 = self.model_s1(x)
        x_s2 = self.model_s2(x)
        # pdb.set_trace()
        x = torch.cat([x_s1, x_s2], dim=1)  # cat along channel axis
        x = self.model_combine(x)
        out_albedo = self.model_albedo(x)
        out_shading = self.model_shading(x)

        return out_shading, out_albedo


##################################################################################
# GAN Blocks: discriminator
##################################################################################
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=1, n_layers=5, pad_type='zero', activ='relu', norm='in'):
        """ Patch-wise discriminator, Fully convolution net
            output_dim: [0,1] ~ [fake, real]
            n_layer: defines the number of max-pooling
        """
        super(Discriminator, self).__init__()
        dim = [input_dim, 16, 32, 32, 64, 64]
        ks = [5, 5, 3, 3, 3]

        self.model = []
        if n_layers <= 5:
            dim = dim[:n_layers+1]
            ks = ks[:n_layers]
        else:
            raise NotImplementedError('Discriminator ' + 'should have layer 5')
        # down sample procedure
        for i in range(n_layers):
            ch_in = dim[i]
            ch_out = dim[i+1]
            kernel_size = ks[i]
            pad_size = (kernel_size - 1) // 2
            self.model += [Conv2dBlock(ch_in, ch_out, kernel_size=kernel_size, padding=pad_size,
                                       pad_type=pad_type, activation=activ, norm=norm)]
            self.model += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # patch discrimination
        self.model += [Conv2dBlock(dim[-1], dim[-1], kernel_size=1,
                                   padding=0, norm='in', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim[-1], output_dim, kernel_size=1,
                                   padding=0, norm='none', activation='sigmoid', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and \
            (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
        # if hasattr(m, 'weight') and (classname.find('ConvTranspose') != -1):
        #     init.constant_(m.weight.data, 1.0)
        #     if hasattr(m, 'bias') and m.bias is not None:
        #         init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, use_drop=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.use_drop = use_drop  # bool, default False
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if self.use_drop:
            self.dropout = nn.Dropout(0.5, True)
        else:
            self.dropout = None

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


class DeConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, out_padding=0, norm='none', activation='relu', pad_type='zero', use_drop=False):
        super(DeConv2dBlock, self).__init__()
        self.use_bias = True
        self.use_drop = use_drop
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if self.use_drop:
            self.dropout = nn.Dropout(0.5, True)
        else:
            self.dropout = None

        # initialize convolution
        self.deconv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride,
                                         output_padding=out_padding, bias=self.use_bias)

    def forward(self, x):
        x = self.deconv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.use_drop:
            x = self.dropout(x)
        return x


# define a resnet(dilated) block
class ResDilateBlock(nn.Module):
    def __init__(self, input_dim, dim, output_dim, rate,
                 padding_type, norm, use_bias=False):
        super(ResDilateBlock, self).__init__()
        feature_, conv_block = self.build_conv_block(input_dim, dim, output_dim, rate,
                                                     padding_type, norm, use_bias)
        self.feature_ = feature_
        self.conv_block = conv_block
        self.activation = nn.ReLU(True)

    def build_conv_block(self, input_dim, dim, output_dim, rate,
                         padding_type, norm, use_bias=False):

        # branch feature_: in case the output_dim is different from input
        feature_ = [self.pad_layer(padding_type, padding=0),
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1,
                              bias=False, dilation=1),
                    self.norm_layer(norm, output_dim),
                    ]
        feature_ = nn.Sequential(*feature_)

        # branch convolution:
        conv_block = []

        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(input_dim, dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, dim),
                       nn.ReLU(True)]
        # dilated conv, padding = dilation_rate, when k=3, s=1, p=d
        # k=5, s=1, p=2d
        conv_block += [self.pad_layer(padding_type='reflect', padding=1*rate),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 bias=False, dilation=rate),
                       self.norm_layer(norm, dim),
                       nn.ReLU(True)]
        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(dim, output_dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, output_dim),
                       ]
        conv_block = nn.Sequential(*conv_block)
        return feature_, conv_block

    @staticmethod
    def pad_layer(padding_type, padding):
        if padding_type == 'reflect':
            pad = nn.ReflectionPad2d(padding)
        elif padding_type == 'replicate':
            pad = nn.ReplicationPad2d(padding)
        elif padding_type == 'zero':
            pad = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return pad

    @staticmethod
    def norm_layer(norm, norm_dim):
        if norm == 'bn':
            norm_layer_ = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            norm_layer_ = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            norm_layer_ = LayerNorm(norm_dim)
        elif norm == 'none':
            norm_layer_ = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        return norm_layer_

    def forward(self, x):
        feature_ = self.feature_(x)
        conv = self.conv_block(x)
        out = feature_ + conv
        out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def _get_norm_layer(norm_type='instance'):
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


##################################################################################
# Distribution distance measurements and losses blocks
##################################################################################

class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.eps = 1e-12
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        return self.kld(x, y)


class Grad_Img(nn.Module):

    def __init__(self, Lambda=0.3):
        """ input image has channel 3 (rgb / bgr)"""
        super(Grad_Img, self).__init__()
        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        #conv_x = np.array([[-1.0,0,1], [-2,0,2], [-1,0,1]])
        conv_x = np.array([[0., 0., 0.], [-1, 0, 1], [0., 0., 0.]], dtype=np.float32)
        kernel_x = np.zeros([3,3,3,3], dtype=np.float32)  # [out_ch,in_ch,ks,ks]
        kernel_x[0, 0, :, :] = conv_x
        kernel_x[1, 1, :, :] = conv_x
        kernel_x[2, 2, :, :] = conv_x
        #conv_y = np.array([[-1.0,-2,-1], [0,0,0], [1,2,1]])
        conv_y = np.array([[0.,-1, 0.], [0, 0, 0], [0., 1, 0.]], dtype=np.float32)
        kernel_y = np.zeros([3, 3, 3, 3], dtype=np.float32)
        kernel_y[0, 0, :, :] = conv_y
        kernel_y[1, 1, :, :] = conv_y
        kernel_y[2, 2, :, :] = conv_y
        self.conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float(), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float(), requires_grad=False)

        # pdb.set_trace()
        # self.Lambda = Lambda

    def forward(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        h, w = input.shape[2], input.shape[3]
        grd_x[:, :, [0, h - 1], :] = 0
        grd_x[:, :, :, [0, w - 1]] = 0
        grd_y[:, :, [0, h - 1], :] = 0
        grd_y[:, :, :, [0, w - 1]] = 0

        out = torch.sqrt(grd_x**2 + grd_y**2) / 2

        return out, grd_x, grd_y


class Grad_Img_v1(nn.Module):

    def __init__(self, Lambda=0.3):
        """ input image has channel 3 (rgb / bgr)
            output gradient images of 1 channel (mean along colour channel)
        """
        super(Grad_Img_v1, self).__init__()
        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x = np.array([[-1.0,0,1], [-2,0,2], [-1,0,1]])
        kernel_x = np.zeros([3,3,3,3])
        kernel_x[0, 0, :, :] = conv_x
        kernel_x[1, 1, :, :] = conv_x
        kernel_x[2, 2, :, :] = conv_x
        conv_y = np.array([[-1.0,-2,-1], [0,0,0], [1,2,1]])
        kernel_y = np.zeros([3, 3, 3, 3])
        kernel_y[0, 0, :, :] = conv_y
        kernel_y[1, 1, :, :] = conv_y
        kernel_y[2, 2, :, :] = conv_y
        self.conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float(), requires_grad=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float(), requires_grad=False)

        # pdb.set_trace()
        # self.Lambda = Lambda

    def forward(self, input):
        grd_x = self.conv_x(input)
        grd_y = self.conv_y(input)
        h, w = input.shape[2], input.shape[3]
        grd_x[:, :, [0, h - 1], :] = 0
        grd_x[:, :, :, [0, w - 1]] = 0
        grd_y[:, :, [0, h - 1], :] = 0
        grd_y[:, :, :, [0, w - 1]] = 0

        out = torch.sqrt(grd_x**2 + grd_y**2) / 2

        #grad_x = torch.mean(grd_x, dim=1, keepdim=True)  #[batch,channel,w,h]
        #grad_y = torch.mean(grd_y, dim=1, keepdim=True)
        out = torch.mean(out, dim=1, keepdim=True)
        out = out.repeat(1,3,1,1 )
        return out, grd_x, grd_y


##################################################################################
# Test codes
##################################################################################

def test_gradient_img():
    from torch.utils.data import DataLoader
    from utils import tensor2img, show_image
    from configs.intrinsic_DI import opt
    import cv2
    import my_data_RD

    MPI_test_dataset = data.DatasetIdMPI(opt.data, is_train=False)
    test_loader = DataLoader(MPI_test_dataset, batch_size=1, shuffle=False)

    for batch_idx, data in enumerate(test_loader):
        I_i, I_s, I_r = data['I'], data['B'], data['R']
        img_i = tensor2img(I_i)
        img_s = tensor2img(I_s)
        img_r = tensor2img(I_r)
        origin_imgs = cv2.vconcat([img_i[:, :, ::-1], img_s[:, :, ::-1], img_r[:, :, ::-1]])

        get_grad = Grad_Img()


        gradI, gradI_x, gradI_y = get_grad(I_i)
        gradR, gradR_x, gradR_y = get_grad(I_r)
        gradS, gradS_x, gradS_y = get_grad(I_s)
        # pdb.set_trace()

        grad_i = tensor2img(gradI_y)
        grad_s = tensor2img(gradS_y)
        grad_r = tensor2img(gradR_y)
        grad_imgs = cv2.vconcat([grad_i[:, :, ::-1], grad_s[:, :, ::-1], grad_r[:, :, ::-1]])
        if not show_image(cv2.hconcat([origin_imgs, grad_imgs])):
            break


if __name__ == '__main__':
    # test_gen()
    test_gradient_img()
