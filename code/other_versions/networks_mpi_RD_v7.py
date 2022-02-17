import functools

from torch import nn
from torch.autograd import Variable
from torch.autograd import grad as ta_grad
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import json
from skimage.transform import resize
# from configs.intrinsic_mpi_v11 import opt
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
    if 'cross-deconv' in train_mode:
        return CrossGen_Unet(gen_opt)
    elif 'self-sup' in train_mode:
        if 'ms' in train_mode:
            return CrossGenMS(gen_opt)  # use this for MPI data
        else:
            return CrossGen(gen_opt)
    elif 'iiw' in train_mode:
        if 'oneway' in train_mode:
            # return GenMS(gen_opt)  # use this for IIW data
            return CrossGenMSIIW(gen_opt)
        elif 'cross' in train_mode:
            return CrossGenMSIIW(gen_opt)
        else:
            raise NotImplementedError
    elif 'unet' in train_mode:
        return YUnetGen(gen_opt)
    else:
        if 'ms' in train_mode:
            return TwoWayGenMS(gen_opt)
        else:
            return TwoWayGen(gen_opt)


def get_discriminator(dis_opt):
    """Get a discriminator, dis_opt is options for discrimination"""
    # Todo: parameters
    # net = Discriminator(input_dim=dis_opt.input_dim, output_dim=1,
    #                     n_layers=dis_opt.n_layers, pad_type=dis_opt.pad_type,
    #                     activ=dis_opt.activ, norm=dis_opt.norm)
    net = Discriminator_v1(dis_opt=dis_opt)
    return net


def get_manual_criterion(name):
    if 'distance' in name.lower():
        return DistanceLoss()
    else:
        raise NotImplementedError(name + 'should in [distance/...]')


##################################################################################
# Generator
##################################################################################
class CrossGen_Unet(nn.Module):
    """ use conv_transpose in decoder """
    def __init__(self, gen_opt):
        super(CrossGen_Unet, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim1 = gen_opt.output_dim_r
        self.output_dim2 = gen_opt.output_dim_s
        self.encoder_name = gen_opt.encoder_name
        self.decoder_init = gen_opt.decoder_init  # True
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'

        # Feature extractor as Encoder
        self.encoder_a = Vgg11Encoder_Unet(input_dim=self.input_dim, pretrained=self.pretrained)
        self.encoder_b = Vgg11Encoder_Unet(input_dim=self.input_dim, pretrained=self.pretrained)

        decoder_a = Decoder_Unet(input_dim=self.input_dim, output_dim=self.output_dim1,
                                 pad_type=self.pad_type, activ=self.activ, norm=self.norm)
        decoder_b = Decoder_Unet(input_dim=self.input_dim, output_dim=self.output_dim2,
                                 pad_type=self.pad_type, activ=self.activ, norm=self.norm)
        if self.decoder_init:
            self.decoder_a = init_net(decoder_a, 'kaiming')
            self.decoder_b = init_net(decoder_b, 'kaiming')
        else:
            self.decoder_a = decoder_a
            self.decoder_b = decoder_b

    def forward(self, x):
        feat_dict_a = self.encoder_a(x)
        feat_dict_b = self.encoder_b(x)

        out_a = self.decoder_a(x, feat_dict_a)
        out_b = self.decoder_b(x, feat_dict_b)
        return out_a, out_b

    def encode(self, x, name):
        if name == 'a':
            return self.encoder_a(x)
        elif name == 'b':
            return self.encoder_b(x)
        else:
            raise NotImplementedError

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        elif name == 'b':
            return self.decoder_b(x, feats)
        else:
            raise NotImplementedError


class CrossGen(nn.Module):
    """ use bilinear interpolation in decoder """
    def __init__(self, gen_opt):
        super(CrossGen, self).__init__()
        self.dim = gen_opt.dim  # 64
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim1 = gen_opt.output_dim_s
        self.output_dim2 = gen_opt.output_dim_r
        self.encoder_name = gen_opt.encoder_name

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder_a = Vgg11Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                          pretrained=self.pretrained)
            self.encoder_b = Vgg11Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                          pretrained=self.pretrained)
        else:

            self.encoder_a = Vgg19Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                          pretrained=self.pretrained)
            self.encoder_b = Vgg19Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                          pretrained=self.pretrained)

        self.decoder_a = Decoder(self.feature_dim, dim=self.dim, output_dim=self.output_dim1,
                                 n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                 norm=self.norm)
        self.decoder_b = Decoder(self.feature_dim, dim=self.dim, output_dim=self.output_dim2,
                                 n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                 norm=self.norm)
        self.fusion = Conv2dBlock(input_dim=self.feature_dim * 2, output_dim=self.feature_dim, kernel_size=1, stride=1)

    def forward(self, x):
        feat_a = self.encoder_a(x)
        feat_b = self.encoder_b(x)
        # pdb.set_trace()
        if self.mode == 'fusion':
            features_a = torch.cat([x, feat_a, x, feat_b], dim=1)
            features_a = self.fusion(features_a)
            features_b = torch.cat([x, feat_b, x, feat_a], dim=1)
            features_b = self.fusion(features_b)
        elif self.mode == 'minus':
            features_a = torch.cat([x, feat_a - feat_b], dim=1)
            features_b = torch.cat([x, feat_b - feat_a], dim=1)
        elif self.mode == 'direct':
            features_a = torch.cat([x, feat_a], dim=1)
            features_b = torch.cat([x, feat_b], dim=1)

        out_a = self.decoder_a(features_a)
        out_b = self.decoder_b(features_b)
        return out_a, out_b

    def encode(self, x, name):
        if name == 'a':
            return self.encoder_a(x)
        elif name == 'b':
            return self.encoder_b(x)
        else:
            raise NotImplementedError

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x)
        elif name == 'b':
            return self.decoder_b(x)
        else:
            raise NotImplementedError


class TwoWayGen(nn.Module):
    def __init__(self, gen_opt):
        super(TwoWayGen, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim1 = gen_opt.output_dim_r
        self.output_dim2 = gen_opt.output_dim_s
        self.encoder_name = gen_opt.encoder_name

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder = Vgg11Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                        pretrained=self.pretrained)
        elif self.encoder_name == 'vgg19':
            self.encoder = Vgg19Encoder(input_dim=self.input_dim, out_feature_dim=self.feature_dim,
                                        pretrained=self.pretrained)

        self.decoder_a = Decoder(self.feature_dim, dim=self.dim, output_dim=self.output_dim1,
                                 n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                 norm=self.norm)
        self.decoder_b = Decoder(self.feature_dim, dim=self.dim, output_dim=self.output_dim1,
                                 n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                 norm=self.norm)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.cat([features, x], dim=1)
        out_a = self.decoder_a(features)
        out_b = self.decoder_b(features)
        return out_a, out_b

    def encode(self, x, name):
        return self.encoder(x)

    def decode(self, x, name):
        if name == 'a':
            return self.decoder_a(x)
        elif name == 'b':
            return self.decoder_b(x)
        else:
            raise NotImplementedError


class YUnetGen(nn.Module):
    def __init__(self, gen_opt):
        super(YUnetGen, self).__init__()
        self.dim = gen_opt.dim
        self.norm = gen_opt.norm
        self.mode = gen_opt.mode
        self.activ = gen_opt.activ
        self.pad_type = gen_opt.pad_type
        self.n_layers = gen_opt.n_layers
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.vgg_pretrained
        self.feature_dim = gen_opt.feature_dim
        self.output_dim1 = gen_opt.output_dim_r
        self.output_dim2 = gen_opt.output_dim_s
        self.encoder_name = gen_opt.encoder_name

        dim = self.dim
        norm_layer = nn.InstanceNorm2d if self.norm == 'in' else nn.BatchNorm2d
        use_dropout = gen_opt.use_dropout

        # construct unet structure
        unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=dim * 8, outer2_nc=dim * 8, inner_nc=dim * 8,
                                                   input_nc=None, submodule1=None, submodule2=None,
                                                   norm_layer=norm_layer, innermost=True)
        for i in range(self.n_layers - 5):
            unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=dim * 8, outer2_nc=dim * 8, inner_nc=dim * 8,
                                                       input_nc=None,
                                                       submodule1=unet_block.model1, submodule2=unet_block.model2,
                                                       norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=dim * 4, outer2_nc=dim * 4, inner_nc=dim * 8,
                                                   input_nc=None,
                                                   submodule1=unet_block.model1, submodule2=unet_block.model2,
                                                   norm_layer=norm_layer)
        unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=dim * 2, outer2_nc=dim * 2, inner_nc=dim * 4,
                                                   input_nc=None,
                                                   submodule1=unet_block.model1, submodule2=unet_block.model2,
                                                   norm_layer=norm_layer)
        unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=dim, outer2_nc=dim, inner_nc=dim * 2,
                                                   input_nc=None,
                                                   submodule1=unet_block.model1, submodule2=unet_block.model2,
                                                   norm_layer=norm_layer)

        unet_block = UnetTwoWaySkipConnectionBlock(outer1_nc=self.output_dim1, outer2_nc=self.output_dim2, inner_nc=dim,
                                                   input_nc=self.input_dim,
                                                   submodule1=unet_block.model1, submodule2=unet_block.model2,
                                                   outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor):
            self.model.cuda()
        return self.model(input)


class CrossGenMS(CrossGen):
    def __init__(self, gen_opt):
        super(CrossGenMS, self).__init__(gen_opt)
        self.encoder_name = gen_opt.encoder_name
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'
        #self.fd_fuc = DivergenceLoss()
        self.fea_dist = DivergenceLoss(detail_weights=gen_opt.div_detail_dict,
                                       cos_w=gen_opt.fd_cosw,
                                       norm_w=gen_opt.fd_normw)

        self.div_dict = gen_opt.div_detail_dict

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder_a = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
        else:
            self.encoder_a = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder_a = DecoderMS(self.input_dim, dim=self.dim,
                                   output_dim=self.output_dim1,  # s
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm, decoder_mode=self.decoder_mode)
        self.decoder_b = DecoderMS(self.input_dim, dim=self.dim,
                                   output_dim=self.output_dim2,  # r
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm, decoder_mode=self.decoder_mode)
        self.fusion = Conv2dBlock(input_dim=1024, output_dim=512, kernel_size=1, stride=1)

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        elif name == 'b':
            return self.decoder_b(x, feats)
        else:
            raise NotImplementedError

    #def compute_fea_diverse(self, features_a, features_b):
    #    d_s1 = self.fd_dist(features_a['low'], features_b['low'])
    #    d_s2 = self.fd_dist(features_a['mid'], features_b['mid'])
    #    d_s3 = self.fd_dist(features_a['deep'], features_b['deep'])
    #    d_s4 = self.fd_dist(features_a['out'], features_b['out'])
    #    return d_s1, d_s2, d_s3, d_s4

    def compute_fd_loss(self, features_a, features_b):
        # feature divergence loss
        return self.fea_dist(features_a, features_b, self.div_dict)

    def forward(self, x):
        feats_a = self.encoder_a(x)
        feats_b = self.encoder_b(x)
        if self.mode == 'fusion':
            features_a = torch.cat([feats_a['out'], feats_b['out']], dim=1)
            features_a = self.fusion(features_a)
            features_b = torch.cat([feats_b['out'], feats_a['out']], dim=1)
            features_b = self.fusion(features_b)
        elif self.mode == 'minus':
            features_a = torch.cat([feats_a['out'] - feats_b['out']], dim=1)
            features_b = torch.cat([feats_b['out'] - feats_a['out']], dim=1)
        elif self.mode == 'direct':
            features_a = feats_a
            features_b = feats_b
        else:
            raise NotImplementedError

        out_a = self.decode(x, 'a', features_a)
        out_b = self.decode(x, 'b', features_b)
        # d_s1, d_s2, d_s3, d_s4 = self.compute_fea_diverse(feats_a, feats_b)
        fd_loss = self.compute_fd_loss(feats_a, feats_b)
        return out_a, out_b, fd_loss  # , [d_s1, d_s2, d_s3, d_s4]


# for iiw dataset
class GenMS(CrossGen):
    def __init__(self, gen_opt):
        super(GenMS, self).__init__(gen_opt)
        self.encoder_name = gen_opt.encoder_name
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'
        self.sigmoid_func = nn.Sigmoid()

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder_a = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
        else:
            self.encoder_a = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder_a = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim1,
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm, decoder_mode=self.decoder_mode)

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        else:
            raise NotImplementedError

    def forward(self, x):
        feats_a = self.encoder_a(x)
        if self.mode == 'direct':
            features_a = feats_a
        else:
            raise NotImplementedError

        out_a = self.decode(x, 'a', features_a)  # reflectance, 1
        out_a = self.sigmoid_func(out_a)  # normalized into [0,1]

        return out_a


class CrossGenMSIIW(CrossGen):
    def __init__(self, gen_opt):
        super(CrossGenMSIIW, self).__init__(gen_opt)
        self.encoder_name = gen_opt.encoder_name
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'
        self.fd_fuc = DivergenceLoss()
        self.sigmoid_func = nn.Sigmoid()

        # Feature extractor as Encoder
        if self.encoder_name == 'vgg11':
            self.encoder_a = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg11EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
        else:
            self.encoder_a = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)
            self.encoder_b = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder_a = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim1,
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm, decoder_mode=self.decoder_mode)
        self.decoder_b = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim1,
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm, decoder_mode=self.decoder_mode)
        self.fusion = Conv2dBlock(input_dim=1024, output_dim=512, kernel_size=1, stride=1)

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        elif name == 'b':
            return self.decoder_b(x, feats)
        else:
            raise NotImplementedError

    def compute_fea_diverse(self, features_a, features_b):
        d_s1 = self.fd_fuc(features_a['low'], features_b['low'])
        d_s2 = self.fd_fuc(features_a['mid'], features_b['mid'])
        d_s3 = self.fd_fuc(features_a['deep'], features_b['deep'])
        d_s4 = self.fd_fuc(features_a['out'], features_b['out'])
        return d_s1, d_s2, d_s3, d_s4

    def forward(self, x):
        feats_a = self.encoder_a(x)
        feats_b = self.encoder_b(x)
        if self.mode == 'fusion':
            features_a = torch.cat([feats_a['out'], feats_b['out']], dim=1)
            features_a = self.fusion(features_a)
            features_b = torch.cat([feats_b['out'], feats_a['out']], dim=1)
            features_b = self.fusion(features_b)
        elif self.mode == 'minus':
            features_a = torch.cat([feats_a['out'] - feats_b['out']], dim=1)
            features_b = torch.cat([feats_b['out'] - feats_a['out']], dim=1)
        elif self.mode == 'direct':
            features_a = feats_a
            features_b = feats_b
        else:
            raise NotImplementedError

        out_a = self.decode(x, 'a', features_a)  # reflectance
        out_b = self.decode(x, 'b', features_b)  # shading
        out_a = self.sigmoid_func(out_a)  # normalized into [0,1]
        out_b = self.sigmoid_func(out_b)  # normalized into [0,1]
        d_s1, d_s2, d_s3, d_s4 = self.compute_fea_diverse(feats_a, feats_b)
        return out_a, out_b, [d_s1, d_s2, d_s3, d_s4]


class TwoWayGenMS(CrossGen):
    def __init__(self, gen_opt):
        super(TwoWayGenMS, self).__init__(gen_opt)
        # Feature extractor as Encoder
        self.decoder_mode = gen_opt.decoder_mode  # 'Basic'/'Residual'
        self.encoder = Vgg19EncoderMS(input_dim=self.input_dim, pretrained=self.pretrained)

        self.decoder_a = DecoderMS(self.input_dim, dim=self.dim,
                                   output_dim=self.output_dim1,  # s
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm,
                                   decoder_mode=self.decoder_mode)
        self.decoder_b = DecoderMS(self.input_dim, dim=self.dim,
                                   output_dim=self.output_dim2,  # r
                                   n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                   norm=self.norm,
                                   decoder_mode=self.decoder_mode)
        # self.fusion = Conv2dBlock(input_dim=1024, output_dim=512, kernel_size=1, stride=1)

    def decode(self, x, name, feats=None):
        if name == 'a':
            return self.decoder_a(x, feats)
        elif name == 'b':
            return self.decoder_b(x, feats)
        else:
            raise NotImplementedError

    def forward(self, x):
        feats = self.encoder(x)

        out_a = self.decode(x, 'a', feats)
        out_b = self.decode(x, 'b', feats)
        return out_a, out_b  #, feats['out']


##################################################################################
# Encoder and Decoders
##################################################################################

class Vgg11Encoder(nn.Module):
    """output the same feature with and height with input"""

    def __init__(self, input_dim, out_feature_dim, pretrained):
        super(Vgg11Encoder, self).__init__()
        features = list(vgg11(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)
        # in-channels = 64+256+512=832, out_channels - input_dim: merge the origin image outside of encoder
        self.merge_features = nn.Conv2d(832, out_feature_dim - input_dim, kernel_size=1)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        image_in = x.clone()
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        # pdb.set_trace()
        # merge different level features of VGG
        merge_feature_names = ['conv1_1', 'conv3_2', 'conv5_2']
        out_feature = None
        _, c, h, w = image_in.shape
        for mfn in merge_feature_names:
            # resize all the extracted feature maps to img size, and concat them all
            feat = result_dict[mfn]
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            out_feature = torch.cat([out_feature, feat], dim=1) if out_feature is not None else feat

        # pdb.set_trace()
        out_feature = self.merge_features(out_feature)
        return out_feature


class Vgg11Encoder_Unet(nn.Module):
    """output the extracted feature maps"""
    # Vgg encoder with multi-scales

    def __init__(self, input_dim, pretrained):
        super(Vgg11Encoder_Unet,  self).__init__()
        features = list(vgg11(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            's1': result_dict['conv1_1'],
            's2': result_dict['conv2_1'],
            's3': result_dict['conv3_2'],
            's4': result_dict['conv4_2'],
            's5': result_dict['conv5_2']
        }
        return out_feature


class Vgg19Encoder(nn.Module):
    def __init__(self, input_dim, out_feature_dim, pretrained):
        super(Vgg19Encoder, self).__init__()
        features = list(vgg19(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)
        # in-channels = 64+256+512=832 out_channels - input_dim: merge the origin image outside of encoder
        self.merge_features = nn.Conv2d(832, out_feature_dim - input_dim, kernel_size=1)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        image_in = x.clone()
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        # merge different level features of VGG
        merge_feature_names = ['conv1_2', 'conv3_4', 'conv5_4']
        out_feature = None
        _, c, h, w = image_in.shape
        for mfn in merge_feature_names:
            feat = result_dict[mfn]
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=True)
            out_feature = torch.cat([out_feature, feat], dim=1) if out_feature is not None else feat

        out_feature = self.merge_features(out_feature)
        return out_feature


class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, input_dim, pretrained):
        super(Vgg11EncoderMS, self).__init__()
        features = list(vgg11(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'low': result_dict['conv1_1'],
            'mid': result_dict['conv2_1'],
            'deep': result_dict['conv3_2'],
            'out': result_dict['conv5_2']
        }
        return out_feature


class Vgg19EncoderMS(nn.Module):
    def __init__(self, input_dim, pretrained):
        super(Vgg19EncoderMS, self).__init__()
        features = list(vgg19(pretrained=pretrained, in_channels=input_dim).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'low': result_dict['conv1_2'],
            'mid': result_dict['conv2_2'],
            'deep': result_dict['conv3_4'], # conv3_4
            'out': result_dict['conv4_4']   # conv5_4
        }
        return out_feature


class Decoder(nn.Module):
    """ without dilated convolution, can set n_layers=1 """
    def __init__(self, input_dim, dim, output_dim, n_layers, pad_type, activ, norm):
        """output_shape = [H, W, C]"""
        super(Decoder, self).__init__()
        self.model = []

        for i in range(n_layers):
            _dim = input_dim if i == 0 else dim
            self.model += [Conv2dBlock(_dim, dim, kernel_size=3, dilation=2 ** i, padding=2 ** i,
                                       pad_type=pad_type, activation=activ, norm=norm)]

        # use reflection padding in the last conv layer
        # print('Decoder Conv Dim is:', dim)  # 64
        # pdb.set_trace()
        self.model += [Conv2dBlock(dim, dim, kernel_size=1, padding=0, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, kernel_size=1, padding=0, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, kernel_size=1, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Decoder_Unet(nn.Module):
    """ use conv2dtranspose instead of bilinear interpolation """
    # -done: initialize function [OK, not work]
    def __init__(self, input_dim, output_dim, pad_type, activ, norm):
        """output_shape = [H, W, C]"""
        super(Decoder_Unet, self).__init__()
        self.pad_type = pad_type
        self.activ = activ
        self.norm = norm
        self.use_bias = True

        # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        inner_nc = [512, 512, 256, 128, 64]
        outer_nc = [256, 256, 128, 128, 64]
        self.up_0 = nn.Conv2d(512, outer_nc[0], kernel_size=3, stride=1,
                              padding=1, bias=self.use_bias)  # to select features from conv_4
        # activation!
        self.up_1 = nn.ConvTranspose2d(512, outer_nc[0], kernel_size=5, stride=2,
                                       padding=2, output_padding=1, bias=self.use_bias)
        self.up_2 = nn.ConvTranspose2d(inner_nc[1]+outer_nc[0], outer_nc[1], kernel_size=4, stride=2,
                                       padding=1, output_padding=0, bias=self.use_bias)
        self.up_3 = nn.ConvTranspose2d(inner_nc[2]+outer_nc[1], outer_nc[2], kernel_size=3, stride=2,
                                       padding=1, output_padding=1, bias=self.use_bias)
        self.up_4 = nn.ConvTranspose2d(inner_nc[3]+outer_nc[2], outer_nc[3], kernel_size=4, stride=2,
                                       padding=1, output_padding=0, bias=self.use_bias)
        self.up_5 = nn.Conv2d(inner_nc[4]+outer_nc[3], outer_nc[4], kernel_size=3, stride=1,
                              padding=1, bias=self.use_bias)  # to select features

        self.pred_model = []
        self.pred_model += [Conv2dBlock(outer_nc[4]+input_dim, 64, kernel_size=3, padding=1,
                                        norm=norm, activation=activ, pad_type=pad_type)]
        self.pred_model += [Conv2dBlock(64, 64, kernel_size=1, padding=0,
                                        norm=norm, activation=activ, pad_type=pad_type)]
        self.pred_model += [Conv2dBlock(64, output_dim, kernel_size=1,
                                        norm='none', activation='none', pad_type=pad_type)]
        self.pred_model = nn.Sequential(*self.pred_model)

    def forward(self, input_x, feat_dict):
        x_1 = feat_dict['s5']  # 1/16, 512
        x_1 = self.up_1(x_1)   # 1/8, 128
        x_2 = feat_dict['s4']  # 1/8, 512
        # x_2 = self.up_0(x_2)   # 1/8, 128
        x_1 = self.up_2(torch.cat([x_1, x_2], dim=1))  # (1/8, 256)-->(1/4, 64)
        x_2 = feat_dict['s3']  # 1/4, 256
        x_1 = self.up_3(torch.cat([x_1, x_2], dim=1))  # (1/4, 320)-->(1/2, 64)
        x_2 = feat_dict['s2']  # 1/2, 128
        x_1 = self.up_4(torch.cat([x_1, x_2], dim=1))  # (1/2, 192)-->(1, 64)
        x_2 = feat_dict['s1']  # 1, 64
        x_1 = self.up_5(torch.cat([x_1, x_2], dim=1))  # (1, 128)-->(1, 64)
        x = torch.cat([x_1, input_x], dim=1)  # (1, 64+3)

        return self.pred_model(x)


class DecoderMS(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_layers, pad_type, activ, norm, decoder_mode='Basic'):
        """output_shape = [H, W, C]"""
        super(DecoderMS, self).__init__()

        self.fuse_out = Conv2dBlock(512, 256, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_deep = Conv2dBlock(512, 128, kernel_size=3, stride=1,
                                     pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_mid = Conv2dBlock(256, 64, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_low = Conv2dBlock(128, 32, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_input = Conv2dBlock(32 + input_dim, dim, kernel_size=3, stride=1, padding=1,
                                      pad_type=pad_type, activation=activ, norm=norm)
        self.contextual_blocks = []
        # TODO: ResBlock, replace current contextual blocks
        rates = [1,2,3,1,1]
        if n_layers > 5:
            raise NotImplementedError('contextual layer should less or equal to 5')
        if decoder_mode == 'Basic':
            for i in range(n_layers):
                self.contextual_blocks += [Conv2dBlock(dim, dim, kernel_size=3, dilation=rates[i], padding=rates[i],
                                                       pad_type=pad_type, activation=activ, norm=norm)]
        elif decoder_mode == 'Residual':
            for i in range(n_layers):
                self.contextual_blocks += [ResDilateBlock(input_dim=dim, dim=dim, output_dim=dim, rate=rates[i],
                                           padding_type=pad_type, norm=norm)]
        else:
            raise NotImplementedError

        # use reflection padding in the last conv layer
        self.contextual_blocks += [
            Conv2dBlock(dim, dim, kernel_size=3, padding=1, norm='in',
                        activation=activ, pad_type='reflect')]
        self.contextual_blocks += [
            Conv2dBlock(dim, output_dim, kernel_size=1, norm='none', activation='none', pad_type=pad_type)]
        self.contextual_blocks = nn.Sequential(*self.contextual_blocks)

    @staticmethod
    def _fuse_feature(x, feature):
        _, _, h, w = feature.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, input_x, feat_dict):
        x = feat_dict['out']
        x = self.fuse_out(x)
        x = self._fuse_feature(x, feat_dict['deep'])
        x = self.fuse_deep(x)
        x = self._fuse_feature(x, feat_dict['mid'])
        x = self.fuse_mid(x)
        x = self._fuse_feature(x, feat_dict['low'])
        x = self.fuse_low(x)
        x = self._fuse_feature(x, input_x)
        x = self.fuse_input(x)

        x = self.contextual_blocks(x)
        return x


##################################################################################
# Modified VGG
##################################################################################
import torch.utils.model_zoo as model_zoo
import math

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(in_channels=3, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg19(in_channels=3, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        in_channels (int):
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


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


class Discriminator_v1(nn.Module):
    def __init__(self, dis_opt):
        """ Patch-wise discriminator, Fully convolution net
            output_dim: [0,1] ~ [fake, real]
            n_layer: defines the number of max-pooling
        """
        super(Discriminator_v1, self).__init__()

        input_dim = dis_opt.input_dim
        output_dim = 1
        n_layers = dis_opt.n_layers
        pad_type = dis_opt.pad_type
        activ = 'lrelu'
        norm = dis_opt.norm

        self.use_grad = dis_opt.use_grad
        self.gan_type = dis_opt.gan_type
        self.grad_w = dis_opt.grad_w

        dim = [input_dim, 16, 32, 32, 64, 64]
        ks = [5, 3, 3, 3, 3]

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
            self.model += [nn.MaxPool2d(kernel_size=3, stride=2)]

        # patch discrimination

        self.model += [Conv2dBlock(dim[-1], output_dim, kernel_size=1,
                                   padding=0, norm='none', activation='sigmoid', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        out0 = self.forward(input_fake)
        out1 = self.forward(input_real)
        loss = 0.0

        if self.gan_type == 'lsgan':
            loss = torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                              F.binary_cross_entropy(F.sigmoid(out1), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        # Gradient penalty
        grad_loss = 0.0
        if self.use_grad:
            eps = Variable(torch.rand(1), requires_grad=True)
            eps = eps.expand(input_real.size())
            eps = eps.cuda()
            x_tilde = eps * input_real + (1 - eps) * input_fake
            x_tilde = x_tilde.cuda()
            pred_tilde = self.calc_gen_loss(x_tilde)
            gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
                                grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_val = self.grad_w * gradients
            grad_loss = ((grad_val.norm(2, dim=1) - 1) ** 2).mean()

        loss += grad_loss

        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        out0 = self.forward(input_fake)
        loss = 0.0

        if self.gan_type == 'lsgan':
            loss += torch.mean((out0 - 1) ** 2)  # LSGAN
        elif self.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out0.data).to(self.device), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


# class MsImageDis(nn.Module):
#     # copy from YunFei's code
#     # Multi-scale discriminator architecture
#     def __init__(self, dis_opt):
#         super(MsImageDis, self).__init__()
#         self.dim = dis_opt.dim
#         self.norm = dis_opt.norm
#         self.activ = dis_opt.activ
#         self.grad_w = dis_opt.grad_w
#         self.pad_type = dis_opt.pad_type
#         self.gan_type = dis_opt.gan_type
#         self.n_layers = dis_opt.n_layers
#         self.use_grad = dis_opt.use_grad
#         self.input_dim = dis_opt.input_dim
#         self.num_scales = dis_opt.num_scales
#         self.use_wasserstein = dis_opt.use_wasserstein
#         self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
#         self.models = nn.ModuleList()
#         self.sigmoid_func = nn.Sigmoid()
#
#         for _ in range(self.num_scales):
#             cnns = self._make_net()
#             if self.use_wasserstein:
#                 cnns += [nn.Sigmoid()]
#
#             self.models.append(nn.Sequential(*cnns))
#
#     def _make_net(self):
#         dim = self.dim
#         cnn_x = []
#         cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
#         for i in range(self.n_layers - 1):
#             cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
#             dim *= 2
#         cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
#         return cnn_x
#
#     def forward(self, x):
#         output = None
#         for model in self.models:
#             out = model(x)
#             if output is not None:
#                 _, _, h, w = out.shape
#                 output = F.interpolate(output, size=(h, w), mode='bilinear')
#                 output = output + out
#             else:
#                 output = out
#
#             x = self.downsample(x)
#
#         output = output / len(self.models)
#         output = self.sigmoid_func(output)
#
#         return output
#
    # def calc_dis_loss(self, input_fake, input_real):
    #     # calculate the loss to train D
    #     outs0 = self.forward(input_fake)
    #     outs1 = self.forward(input_real)
    #     loss = 0
    #
    #     for it, (out0, out1) in enumerate(zip(outs0, outs1)):
    #         if self.gan_type == 'lsgan':
    #             loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
    #         elif self.gan_type == 'nsgan':
    #             all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
    #             all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
    #             loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
    #                                F.binary_cross_entropy(F.sigmoid(out1), all1))
    #         else:
    #             assert 0, "Unsupported GAN type: {}".format(self.gan_type)
    #
    #         # Gradient penalty
    #         grad_loss = 0
    #         if self.use_grad:
    #             eps = Variable(torch.rand(1), requires_grad=True)
    #             eps = eps.expand(input_real.size())
    #             eps = eps.cuda()
    #             x_tilde = eps * input_real + (1 - eps) * input_fake
    #             x_tilde = x_tilde.cuda()
    #             pred_tilde = self.calc_gen_loss(x_tilde)
    #             gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
    #                                 grad_outputs=torch.ones(pred_tilde.size()).cuda(),
    #                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    #             grad_loss = self.grad_w * gradients
    #
    #             input_real = self.downsample(input_real)
    #             input_fake = self.downsample(input_fake)
    #
    #         loss += ((grad_loss.norm(2, dim=1) - 1) ** 2).mean()
    #
    #     return loss
    #
    # def calc_gen_loss(self, input_fake):
    #     # calculate the loss to train G
    #     outs0 = self.forward(input_fake)
    #     loss = 0
    #     for it, (out0) in enumerate(outs0):
    #         if self.gan_type == 'lsgan':
    #             loss += torch.mean((out0 - 1) ** 2)  # LSGAN
    #         elif self.gan_type == 'nsgan':
    #             all1 = Variable(torch.ones_like(out0.data).to(self.device), requires_grad=False)
    #             loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
    #         else:
    #             assert 0, "Unsupported GAN type: {}".format(self.gan_type)
    #     return loss


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
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
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

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DeConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, out_padding=0, norm='none', activation='relu', pad_type='zero'):
        super(DeConv2dBlock, self).__init__()
        self.use_bias = True
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

        # initialize convolution
        self.deconv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride,
                                         output_padding=out_padding, bias=self.use_bias)

    def forward(self, x):
        x = self.deconv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
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


# Defines the submodule with two-way skip connection.
# X -------------------identity-------------------- X
# |                 /- |submodule1| -- up-sampling1 --|
# |-- down-sampling    |          |                   |
# |                 \- |submodule2| -- up-sampling2 --|
class UnetTwoWaySkipConnectionBlock(nn.Module):
    def __init__(self, outer1_nc, outer2_nc, inner_nc, input_nc=None, submodule1=None, submodule2=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetTwoWaySkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = max(outer1_nc, outer2_nc)
        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                              stride=2, padding=1, bias=use_bias)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc)
        up_relu = nn.ReLU(True)
        up_norm1 = norm_layer(outer1_nc)
        up_norm2 = norm_layer(outer2_nc)

        if outermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [down_conv]
            up1 = [up_relu, upconv1, nn.Tanh()]
            up2 = [up_relu, upconv2, nn.Tanh()]
            model1 = down + [submodule1] + up1
            model2 = down + [submodule2] + up2
        elif innermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv]
            up1 = [up_relu, upconv1, up_norm1]
            up2 = [up_relu, upconv2, up_norm2]
            model1 = down + up1
            model2 = down + up2
        else:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer1_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer2_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [down_relu, down_conv, down_norm]
            up1 = [up_relu, upconv1, up_norm1]
            up2 = [up_relu, upconv2, up_norm2]

            if use_dropout:
                model1 = down + [submodule1] + up1 + [nn.Dropout(0.5)]
                model2 = down + [submodule2] + up1 + [nn.Dropout(0.5)]
            else:
                model1 = down + [submodule1] + up1
                model2 = down + [submodule2] + up2

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, x):
        if self.outermost:
            return self.model1(x), self.model2(x)
        else:
            return torch.cat([x, self.model1(x)], 1), torch.cat([x, self.model2(x)], 1)
        pass


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


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super(JSDivergence, self).__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class DivergenceLoss_bak(nn.Module):
    """assume orthogonal is max divergence"""
    def __init__(self):
        super(DivergenceLoss_bak, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        # pass

    def forward(self, features_x, features_y):
        # normalize
        cos = self.cos_sim(features_x, features_y)  # cosine value between x and y
        d = torch.pow(cos, 2) * 2.0  # symmetry function
        loss = torch.mean(d)
        return loss

class DivergenceLoss(nn.Module):
    """assume orthogonal is max divergence
        Pers_Loss smaller, similarity larger
    """
    def __init__(self, detail_weights, cos_w=0.99, norm_w=0.01, alpha=1.2, scale=1.0):
        super(DivergenceLoss, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.l2_diff = nn.MSELoss()
        self.l1_diff = nn.L1Loss()
        self.detail_weights = detail_weights  # opt.optim.div_detail_dict
        self.scale = scale
        self.alpha = alpha
        self.cos_w = cos_w
        self.norm_w = norm_w
        # pass

    def _compute_dist_loss(self, fea_x, fea_y):
        # x, y shape=[B,C,H,W]
        # pdb.set_trace()
        # for image pool data
        # encourage smaller cosine and larger norm distance
        if fea_x.size(0) < fea_y.size(0):
            n = fea_y.size(0) // fea_x.size(0) + 1
            fea_x = fea_x.repeat(n, 1, 1, 1)[:fea_y.size(0),:,:,:]
        elif fea_x.size(0) > fea_y.size(0):
            n = fea_x.size(0) // fea_y.size(0) + 1
            fea_y = fea_y.repeat(n, 1, 1, 1)[:fea_x.size(0),:,:,:]
        # pdb.set_trace()
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        #d_cos = (1.0 - cos) * 1.0  # cosine distance smaller, d smaller
        d_cos = torch.pow(cos, 2) * 2.0  # symmetry function; the smaller, the better
        d_cos = torch.mean(d_cos)

        d_l2 = self.l1_diff(fea_x, fea_y)  # the smaller, more similar
        d_l2 = self._rescale_distance(d_l2)  # normed into value range (0,1)
        d_l2 = torch.mean(d_l2)

        d = d_cos*self.cos_w + d_l2*self.norm_w
        return d

    def _rescale_distance(self, dist):
        d_ = dist * self.scale
        g_ = -(d_ - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_rescale = (1 / (1 + torch.exp(g_)))
        return 1 - d_rescale  # the larger distance, the better

    def _compute_dist_loss_v1(self, fea_x, fea_y):
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d = (1.0 - torch.pow(cos, 2)) * 2.0
        d = torch.mean(d)
        return d

    def forward(self, features_x, features_y, detail_weights=None):
        fea_weights = self.detail_weights if detail_weights is None else detail_weights
        loss = 0
        n_sum = 0
        for key in features_x.keys():
            loss += self._compute_dist_loss(features_x[key], features_y[key]) * fea_weights[key]
            n_sum += 1

        loss = loss / (n_sum + self.eps)

        return loss


class DistanceLoss_YF(nn.Module):
    """assume a*b==-1 is max divergence"""
    def __init__(self):
        super(DistanceLoss_YF, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()  # why need this?---speed up
        # pass

    def forward(self, features_x, features_y):
        # normalize
        d = self.cos_sim(features_x, features_y)  # cosine value between x and y
        d = self.sigmoid(d)
        d = -torch.log(torch.abs(d))
        # d = 1 - torch.abs(d)
        loss = torch.mean(d)
        return loss


class DistanceLoss(nn.Module):
    def __init__(self, alpha=1.4, scale=1.):
        """
        DistanceLoss
        :param alpha: see eq. 6
        :param scale: scale the L1 distance between two image features
        """
        super(DistanceLoss, self).__init__()
        self.eps = 1e-12
        self.alpha = alpha
        self.scale = scale
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()
        self.l1_diff = nn.L1Loss()
        pass

    def _compute_dist_loss(self, dist):
        # normalize
        d_l1 = dist * self.scale
        g_ab = -(d_l1 - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_psi = 1 / (1 + torch.exp(g_ab))

        return 1 - d_psi

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self._compute_dist_loss(self.l1_diff(features_x[key], features_y[key]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self._compute_dist_loss(self.l1_diff(features_x[idx], features_y[idx]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)

        else:
            loss = self._compute_dist_loss(self.l1_diff(features_x, features_y))

        return loss


class PerspectiveLoss(nn.Module):
    """assume orthogonal is max divergence
        Pers_Loss smaller, similarity larger
    """
    def __init__(self, detail_weights, cos_w=0.01, norm_w=0.99, alpha=1.2, scale=1.0):
        super(PerspectiveLoss, self).__init__()
        self.eps = 1e-12
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.l2_diff = nn.MSELoss()
        self.l1_diff = nn.L1Loss()
        self.detail_weights = detail_weights # opt.optim.div_detail_dict_equal
        self.scale = scale
        self.alpha = alpha
        self.cos_w = cos_w
        self.norm_w = norm_w
        # pass

    def _compute_dist_loss(self, fea_x, fea_y):
        # x, y shape=[B,C,H,W]
        # pdb.set_trace()
        # for image pool data
        # encourage larger cosine and smaller norm distance
        if fea_x.size(0) < fea_y.size(0):
            n = fea_y.size(0) // fea_x.size(0) + 1
            fea_x = fea_x.repeat(n, 1, 1, 1)[:fea_y.size(0),:,:,:]
        elif fea_x.size(0) > fea_y.size(0):
            n = fea_x.size(0) // fea_y.size(0) + 1
            fea_y = fea_y.repeat(n, 1, 1, 1)[:fea_x.size(0),:,:,:]
        # pdb.set_trace()
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d_cos = (1.0 - cos) * 1.0  # cosine distance smaller, d smaller
        d_cos = torch.mean(d_cos)

        d_l2 = self.l1_diff(fea_x, fea_y)  # the smaller, more similar
        d_l2 = self._rescale_distance(d_l2)  # normed into value range (0,1)
        d_l2 = torch.mean(d_l2)

        d = d_cos*self.cos_w + d_l2*self.norm_w
        return d

    def _rescale_distance(self, dist):
        d_ = dist * self.scale
        g_ = -(d_ - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_rescale = 1 / (1 + torch.exp(g_))
        return d_rescale

    def _compute_dist_loss_v1(self, fea_x, fea_y):
        cos = self.cos_sim(fea_x, fea_y)  # cosine value between x and y
        d = (1.0 - torch.pow(cos, 2)) * 2.0
        d = torch.mean(d)
        return d

    def forward(self, features_x, features_y, detail_weights=None):
        fea_weights = self.detail_weights if detail_weights is None else detail_weights
        loss = 0
        n_sum = 0
        for key in features_x.keys():
            loss += self._compute_dist_loss(features_x[key], features_y[key]) * fea_weights[key]
            n_sum += 1

        loss = loss / (n_sum + self.eps)

        return loss


class PerspectiveLoss_v0(nn.Module):
    def __init__(self, alpha=1.4, scale=1.):
        """
        DistanceLoss
        :param alpha: see eq. 6
        :param scale: scale the L1 distance between two image features
        """
        super(PerspectiveLoss_v0, self).__init__()
        self.eps = 1e-12
        self.alpha = alpha
        self.scale = scale
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=self.eps)
        self.sigmoid = nn.Sigmoid()
        self.l1_diff = nn.L1Loss()
        pass

    def _compute_dist_loss(self, dist):
        # normalize
        d_l1 = dist * self.scale
        g_ab = -(d_l1 - self.alpha * np.exp(self.alpha)) / (self.alpha ** 2)
        d_psi = 1 / (1 + torch.exp(g_ab))

        return d_psi

    def forward(self, features_x, features_y):
        loss = 0
        n_sum = 0
        if isinstance(features_x, dict):
            for key in features_x.keys():
                loss += self._compute_dist_loss(self.l1_diff(features_x[key], features_y[key]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)
        elif isinstance(features_x, list):
            for idx in range(len(features_x)):
                loss += self._compute_dist_loss(self.l1_diff(features_x[idx], features_y[idx]))
                n_sum += 1

            loss = loss / (n_sum + self.eps)

        else:
            loss = self._compute_dist_loss(self.l1_diff(features_x, features_y))

        return loss


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        """Patch discriminator, input: (B,C,H,W)"""
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(size=input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(size=input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)


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


# Joint Loss for IIW data training
class Sparse(Function):
    # Sparse matrix for S
    def forward(self, input, S):
        self.save_for_backward(S)
        output = torch.mm(S, input)
        # output = output.cuda()
        return output

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        S,  = self.saved_tensors
        grad_weight  = None
        grad_input = torch.mm(S.t(), grad_output)
        # grad_input = grad_input.cuda()
        return grad_input, grad_weight


class JointLoss(nn.Module):
    def __init__(self, opt_optim):
        super(JointLoss, self).__init__()
        self.w_rs_local = opt_optim.w_rs_local
        self.w_reconstr_real = opt_optim.w_reconstr_real
        self.w_rs_dense = 2.0
        self.w_ss_dense = opt_optim.w_ss_dense
        self.w_IIW = opt_optim.w_IIW
        self.w_grad = 0.25

        self.Tensor = torch.cuda.FloatTensor

        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        # self.h_offset = [0,0,0,1,1,2,2,2,1]
        # self.w_offset = [0,1,2,0,2,0,1,2,1]
        self.total_loss = None
        self.running_stage = 0

    def BilateralRefSmoothnessLoss(self, pred_R, targets, att, num_features):
        # pred_R = pred_R.cpu()
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        N = pred_R.size(2) * pred_R.size(3)
        Z = (pred_R.size(1) * N)

        # grad_input = torch.FloatTensor(pred_R.size())
        # grad_input = grad_input.zero_()

        for i in range(pred_R.size(0)):  # for each image
            B_mat = targets[att + 'B_list'][i]  # still list of blur sparse matrices
            S_mat = Variable(targets[att + 'S'][i].cuda(), requires_grad=False)  # Splat and Slicing matrix
            n_vec = Variable(targets[att + 'N'][i].cuda(),
                             requires_grad=False)  # bi-stochatistic vector, which is diagonal matrix

            p = pred_R[i, :, :, :].view(pred_R.size(1), -1).t()  # NX3
            # p'p
            # p_norm = torch.mm(p.t(), p)
            # p_norm_sum = torch.trace(p_norm)
            p_norm_sum = torch.sum(torch.mul(p, p))

            # S * N * p
            Snp = torch.mul(n_vec.repeat(1, pred_R.size(1)), p)
            sp_mm = Sparse()
            Snp = sp_mm(Snp, S_mat)

            Snp_1 = Snp.clone()
            Snp_2 = Snp.clone()

            # # blur
            for f in range(num_features + 1):
                B_var1 = Variable(B_mat[f].cuda(), requires_grad=False)
                sp_mm1 = Sparse()
                Snp_1 = sp_mm1(Snp_1, B_var1)

                B_var2 = Variable(B_mat[num_features - f].cuda(), requires_grad=False)
                sp_mm2 = Sparse()
                Snp_2 = sp_mm2(Snp_2, B_var2)

            Snp_12 = Snp_1 + Snp_2
            pAp = torch.sum(torch.mul(Snp, Snp_12))

            total_loss = total_loss + ((p_norm_sum - pAp) / Z)

        total_loss = total_loss / pred_R.size(0)
        # average over all images
        return total_loss

    def IIWReconstLoss(self, R, S, targets):
        S = S.repeat(1, 3, 1, 1)
        rgb_img = Variable(targets['rgb_img'].cuda(), requires_grad=False)

        # 1 channel
        chromaticity = Variable(targets['chromaticity'].cuda(), requires_grad=False)
        p_R = torch.mul(chromaticity, R.repeat(1, 3, 1, 1))

        # return torch.mean( torch.mul(L, torch.pow( torch.log(rgb_img) - torch.log(p_R) - torch.log(S), 2)))
        return torch.mean(torch.pow(rgb_img - torch.mul(p_R, S), 2))

    def Ranking_Loss(self, prediction_R, judgements, is_flip):
        # ranking loss for each prediction feature
        tau = 0.25  # abs(I1 - I2)) ) #1.2 * (1 + math.fabs(math.log(I1) - math.log(I2) ) )

        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)

        num_valid_comparisons = 0

        num_valid_comparisons_ineq = 0
        num_valid_comparisons_eq = 0

        total_loss_eq = Variable(torch.cuda.FloatTensor(1))
        total_loss_eq[0] = 0
        total_loss_ineq = Variable(torch.cuda.FloatTensor(1))
        total_loss_ineq[0] = 0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            #  remove unconfident point
            weight = c['darker_score']
            if weight < 0.5 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]

            if not point1['opaque'] or not point2['opaque']:
                continue

            # if is_flip:
            # l1 = prediction_R[:, int(point1['y'] * rows), cols - 1 - int( point1['x'] * cols)]
            # l2 = prediction_R[:, int(point2['y'] * rows), cols - 1 - int( point2['x'] * cols)]
            # else:
            l1 = prediction_R[:, int(point1['y'] * rows), int(point1['x'] * cols)]
            l2 = prediction_R[:, int(point2['y'] * rows), int(point2['x'] * cols)]

            l1_m = l1  # torch.mean(l1)
            l2_m = l2  # torch.mean(l2)

            # print(int(point1['y'] * rows), int(point1['x'] * cols), int(point2['y'] * rows), int(point2['x'] * cols), darker)
            # print(point1['y'], point1['x'], point2['y'], point2['x'], c['point1'], c['point2'])
            # print("===============================================================")
            # l2 > l1, l2 is brighter
            # if darker == '1' and ((l1_m.data[0] / l2_m.data[0]) > 1.0/tau):
            #     # loss =0
            #     loss =  weight * torch.mean((tau -  (l2_m / l1_m)))
            #     num_valid_comparisons += 1
            # # l1 > l2, l1 is brighter
            # elif darker == '2' and ((l2_m.data[0] / l1_m.data[0]) > 1.0/tau):
            #     # loss =0
            #     loss =  weight * torch.mean((tau -  (l1_m / l2_m)))
            #     num_valid_comparisons += 1
            # # is equal
            # elif darker == 'E':
            #     loss =  weight * torch.mean(torch.abs(l2 - l1))
            #     num_valid_comparisons += 1
            # else:
            #     loss = 0.0

            # l2 is brighter
            if darker == '1' and ((l1_m.data[0] - l2_m.data[0]) > - tau):
                # print("dark 1", l1_m.data[0] - l2_m.data[0])
                total_loss_ineq += weight * torch.mean(torch.pow(tau - (l2_m - l1_m), 2))
                num_valid_comparisons_ineq += 1.
                # print("darker 1 loss", l2_m.data[0], l1_m.data[0], loss.data[0])
            # l1 > l2, l1 is brighter
            elif darker == '2' and ((l2_m.data[0] - l1_m.data[0]) > - tau):
                # print("dark 2", l2_m.data[0] - l1_m.data[0])
                total_loss_ineq += weight * torch.mean(torch.pow(tau - (l1_m - l2_m), 2))
                num_valid_comparisons_ineq += 1.
                # print("darker 2 loss", l2_m.data[0], l1_m.data[0], loss.data[0])
            elif darker == 'E':
                total_loss_eq += weight * torch.mean(torch.pow(l2 - l1, 2))
                num_valid_comparisons_eq += 1.
            else:
                loss = 0.0

        total_loss = total_loss_ineq + total_loss_eq
        num_valid_comparisons = num_valid_comparisons_eq + num_valid_comparisons_ineq

        # print("average eq loss", total_loss_eq.data[0]/(num_valid_comparisons_eq + 1e-6))
        # print("average ineq loss", total_loss_ineq.data[0]/(num_valid_comparisons_ineq + 1e-6))

        return total_loss / (num_valid_comparisons + 1e-6)

    def BatchRankingLoss(self, prediction_R, judgements_eq, judgements_ineq, random_filp):
        eq_loss, ineq_loss = 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0
        tau = 0.425

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)
        num_channel = prediction_R.size(0)

        # evaluate equality annotations densely
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()
            R_vec = prediction_R.view(num_channel, -1)
            # R_vec = torch.exp(R_vec)
            # I_vec = I.view(1, -1)

            y_1 = torch.floor(judgements_eq[:, 0] * rows).long()
            y_2 = torch.floor(judgements_eq[:, 2] * rows).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_eq[:, 1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_eq[:, 3] * cols).long()
            else:
                x_1 = torch.floor(judgements_eq[:, 1] * cols).long()
                x_2 = torch.floor(judgements_eq[:, 3] * cols).long()

            # compute linear index for point 1
            # y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_eq[:,1] * cols).long()
            point_1_idx_linaer = y_1 * cols + x_1
            # compute linear index for point 2
            # y_2 = torch.floor(judgements_eq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_eq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec, 1, Variable(point_1_idx_linaer, requires_grad=False))
            points_2_vec = torch.index_select(R_vec, 1, Variable(point_2_idx_linear, requires_grad=False))

            # I1_vec = torch.index_select(I_vec, 1, point_1_idx_linaer)
            # I2_vec = torch.index_select(I_vec, 1, point_2_idx_linear)

            weight = Variable(judgements_eq[:, 4], requires_grad=False)
            # weight = confidence#* torch.exp(4.0 * torch.abs(I1_vec - I2_vec) )

            # compute loss
            # eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.abs(points_1_vec - points_2_vec),0) ))
            eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.pow(points_1_vec - points_2_vec, 2), 0)))
            num_valid_eq += judgements_eq.size(0)

            # compute inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            R_intensity = torch.mean(prediction_R, 0)
            # R_intensity = torch.log(R_intensity)
            R_vec_mean = R_intensity.view(1, -1)

            y_1 = torch.floor(judgements_ineq[:, 0] * rows).long()
            y_2 = torch.floor(judgements_ineq[:, 2] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_ineq[:, 1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_ineq[:, 3] * cols).long()
            else:
                x_1 = torch.floor(judgements_ineq[:, 1] * cols).long()
                x_2 = torch.floor(judgements_ineq[:, 3] * cols).long()

            # y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            point_1_idx_linaer = y_1 * cols + x_1
            # y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec_mean, 1, Variable(point_1_idx_linaer, requires_grad=False)).squeeze(
                0)
            points_2_vec = torch.index_select(R_vec_mean, 1, Variable(point_2_idx_linear, requires_grad=False)).squeeze(
                0)
            weight = Variable(judgements_ineq[:, 4], requires_grad=False)

            # point 2 should be always darker than (<) point 1
            # compute loss
            relu_layer = nn.ReLU(True)
            # ineq_loss = torch.sum(torch.mul(weight, relu_layer(points_2_vec - points_1_vec + tau) ) )
            ineq_loss = torch.sum(torch.mul(weight, torch.pow(relu_layer(points_2_vec - points_1_vec + tau), 2)))
            # ineq_loss = torch.sum(torch.mul(weight, torch.pow(relu_layer(tau - points_1_vec/points_2_vec),2)))

            num_included = torch.sum(torch.ge(points_2_vec.data - points_1_vec.data, -tau).float().cuda())
            # num_included = torch.sum(torch.ge(points_2_vec.data/points_1_vec.data, 1./tau).float().cuda())

            num_valid_ineq += num_included

            # avoid divide by zero
        return eq_loss / (num_valid_eq + 1e-8) + ineq_loss / (num_valid_ineq + 1e-8)

    def ShadingPenaltyLoss(self, S):
        return torch.mean(torch.pow(S - 0.5, 2))
        # return torch.sum( torch.mul(sky_mask, torch.abs(S - np.log(0.5))/num_val_pixels ))

    def AngleLoss(self, prediction_n, targets):
        mask = Variable(targets['mask'].cuda(), requires_grad=False)
        normal = Variable(targets['normal'].cuda(), requires_grad=False)
        num_valid = torch.sum(mask[:, 0, :, :])
        # compute dot product
        angle_loss = - torch.sum(torch.mul(mask, torch.mul(prediction_n, normal)), 1)
        return 1 + torch.sum(angle_loss) / num_valid

    def GradientLoss(self, prediction_n, mask, gt_n):
        N = torch.sum(mask)

        # horizontal angle difference
        h_mask = torch.mul(mask[:, :, :, 0:-2], mask[:, :, :, 2:])
        h_gradient = prediction_n[:, :, :, 0:-2] - prediction_n[:, :, :, 2:]
        h_gradient_gt = gt_n[:, :, :, 0:-2] - gt_n[:, :, :, 2:]
        h_gradient_loss = torch.mul(h_mask, torch.abs(h_gradient - h_gradient_gt))

        # Vertical angle difference
        v_mask = torch.mul(mask[:, :, 0:-2, :], mask[:, :, 2:, :])
        v_gradient = prediction_n[:, :, 0:-2, :] - prediction_n[:, :, 2:, :]
        v_gradient_gt = gt_n[:, :, 0:-2, :] - gt_n[:, :, 2:, :]
        v_gradient_loss = torch.mul(v_mask, torch.abs(v_gradient - v_gradient_gt))

        gradient_loss = torch.sum(h_gradient_loss) + torch.sum(v_gradient_loss)
        gradient_loss = gradient_loss / (N * 2.0)

        return gradient_loss

    def SmoothLoss(self, prediction_n, mask):
        N = torch.sum(mask[:, 0, :, :])

        # horizontal angle difference
        h_mask = torch.mul(mask[:, :, :, 0:-2], mask[:, :, :, 2:])
        h_gradient = torch.sum(torch.mul(h_mask, torch.mul(prediction_n[:, :, :, 0:-2], prediction_n[:, :, :, 2:])), 1)
        h_gradient_loss = 1 - torch.sum(h_gradient) / N

        # Vertical angle difference
        v_mask = torch.mul(mask[:, :, 0:-2, :], mask[:, :, 2:, :])
        v_gradient = torch.sum(torch.mul(v_mask, torch.mul(prediction_n[:, :, 0:-2, :], prediction_n[:, :, 2:, :])), 1)
        v_gradient_loss = 1 - torch.sum(v_gradient) / N

        gradient_loss = h_gradient_loss + v_gradient_loss

        return gradient_loss

    def UncertaintyLoss(self, prediction_n, uncertainty, targets):
        uncertainty = torch.squeeze(uncertainty, 1)

        mask = Variable(targets['mask'].cuda(), requires_grad=False)
        normal = Variable(targets['normal'].cuda(), requires_grad=False)
        num_valid = torch.sum(mask[:, 0, :, :])

        angle_diff = (torch.sum(torch.mul(prediction_n, normal), 1) + 1.0) * 0.5
        uncertainty_loss = torch.sum(torch.mul(mask[:, 0, :, :], torch.pow(uncertainty - angle_diff, 2)))
        return uncertainty_loss / num_valid

    def MaskLocalSmoothenessLoss(self, R, M, targets):
        h = R.size(2)
        w = R.size(3)
        num_c = R.size(1)

        half_window_size = 1
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        mask_center = M[:, :,
                      half_window_size + self.Y[half_window_size, half_window_size]:h - half_window_size + self.Y[
                          half_window_size, half_window_size], \
                      half_window_size + self.X[half_window_size, half_window_size]:w - half_window_size + self.X[
                          half_window_size, half_window_size]]

        R_center = R[:, :, half_window_size + self.Y[half_window_size, half_window_size]:h - half_window_size + self.Y[
            half_window_size, half_window_size], \
                   half_window_size + self.X[half_window_size, half_window_size]:w - half_window_size + self.X[
                       half_window_size, half_window_size]]

        c_idx = 0

        for k in range(0, half_window_size * 2 + 1):
            for l in range(0, half_window_size * 2 + 1):
                # albedo_weights = Variable(targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,num_c,1,1).float().cuda(), requires_grad = False)
                R_N = R[:, :, half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                      half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]
                mask_N = M[:, :, half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                         half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]

                composed_M = torch.mul(mask_N, mask_center)

                # albedo_weights = torch.mul(albedo_weights, composed_M)

                r_diff = torch.mul(composed_M, torch.pow(R_center - R_N, 2))
                total_loss = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1

        return total_loss / (8.0 * num_c)

    def LocalAlebdoSmoothenessLoss(self, R, targets, scale_idx):
        h = R.size(2)
        w = R.size(3)
        num_c = R.size(1)

        half_window_size = 1
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        R_center = R[:, :, half_window_size + self.Y[half_window_size, half_window_size]:h - \
                           half_window_size + self.Y[half_window_size, half_window_size],
                           half_window_size + self.X[half_window_size, half_window_size]:w - \
                           half_window_size + self.X[half_window_size, half_window_size]]

        c_idx = 0

        for k in range(0, half_window_size * 2 + 1):
            for l in range(0, half_window_size * 2 + 1):
                albedo_weights = targets["r_w_s" + str(scale_idx)][:, c_idx, :, :].unsqueeze(1).repeat(1, num_c, 1,
                                                                                                       1).float().cuda()
                R_N = R[:, :, half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                        half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]
                # mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
                # composed_M = torch.mul(mask_N, mask_center)
                # albedo_weights = torch.mul(albedo_weights, composed_M)
                r_diff = torch.mul(Variable(albedo_weights, requires_grad=False), torch.abs(R_center - R_N))

                total_loss = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1

        return total_loss / (8.0 * num_c)

    def Data_Loss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)
        s1 = torch.sum(torch.pow(log_diff, 2)) / N
        s2 = torch.pow(torch.sum(log_diff), 2) / (N * N)
        data_loss = s1 - s2
        return data_loss

    def L2GradientMatchingLoss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)

        v_gradient = torch.pow(log_diff[:, :, 0:-2, :] - log_diff[:, :, 2:, :], 2)
        v_mask = torch.mul(mask[:, :, 0:-2, :], mask[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.pow(log_diff[:, :, :, 0:-2] - log_diff[:, :, :, 2:], 2)
        h_mask = torch.mul(mask[:, :, :, 0:-2], mask[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))
        gradient_loss = gradient_loss / N

        return gradient_loss

    def L1GradientMatchingLoss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)

        v_gradient = torch.abs(log_diff[:, :, 0:-2, :] - log_diff[:, :, 2:, :])
        v_mask = torch.mul(mask[:, :, 0:-2, :], mask[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_diff[:, :, :, 0:-2] - log_diff[:, :, :, 2:])
        h_mask = torch.mul(mask[:, :, :, 0:-2], mask[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / 2.0
        gradient_loss = gradient_loss / N

        return gradient_loss

    def L1Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum(mask)
        diff = torch.mul(mask, torch.abs(prediction_n - gt))
        return torch.sum(diff) / num_valid

    def L2Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum(mask)

        diff = torch.mul(mask, torch.pow(prediction_n - gt, 2))
        return torch.sum(diff) / num_valid

    def HuberLoss(self, prediction, mask, gt):
        tau = 1.0
        num_valid = torch.sum(mask)

        diff_L1 = torch.abs(prediction - gt)
        diff_L2 = torch.pow(prediction - gt, 2)

        mask_L2 = torch.le(diff_L1, tau).float().cuda()
        mask_L1 = 1.0 - mask_L2

        L2_loss = 0.5 * torch.sum(torch.mul(mask, torch.mul(mask_L2, diff_L2)))
        L1_loss = torch.sum(torch.mul(mask, torch.mul(mask_L1, diff_L1))) - 0.5

        final_loss = (L2_loss + L1_loss) / num_valid
        return final_loss

    def CCLoss(self, prediction_S, saw_mask, gts, num_cc):
        diff = prediction_S - gts
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        num_regions = 0

        # for each prediction
        for i in range(prediction_S.size(0)):
            log_diff = diff[i, :, :, :]
            mask = saw_mask[i, :, :, :].int()

            for k in range(1, num_cc[i] + 1):
                new_mask = (mask == k).float().cuda()

                masked_log_diff = torch.mul(new_mask, log_diff)
                N = torch.sum(new_mask)

                s1 = torch.sum(torch.pow(masked_log_diff, 2)) / N
                s2 = torch.pow(torch.sum(masked_log_diff), 2) / (N * N)
                total_loss += (s1 - s2)
                num_regions += 1

        return total_loss / (num_regions + 1e-6)

    def DirectFramework(self, prediction, gt, mask):

        w_data = 1.0
        w_grad = 0.5
        final_loss = w_data * self.L2Loss(prediction, mask, gt)

        # level 0
        prediction_1 = prediction[:, :, ::2, ::2]
        prediction_2 = prediction_1[:, :, ::2, ::2]
        prediction_3 = prediction_2[:, :, ::2, ::2]

        mask_1 = mask[:, :, ::2, ::2]
        mask_2 = mask_1[:, :, ::2, ::2]
        mask_3 = mask_2[:, :, ::2, ::2]

        gt_1 = gt[:, :, ::2, ::2]
        gt_2 = gt_1[:, :, ::2, ::2]
        gt_3 = gt_2[:, :, ::2, ::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction, mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    # all parameter in log space, presumption
    def ScaleInvarianceFramework(self, prediction, gt, mask, w_grad):

        assert (prediction.size(1) == gt.size(1))
        assert (prediction.size(1) == mask.size(1))

        w_data = 1.0
        final_loss = w_data * self.Data_Loss(prediction, mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction, mask, gt)

        # level 0
        prediction_1 = prediction[:, :, ::2, ::2]
        prediction_2 = prediction_1[:, :, ::2, ::2]
        prediction_3 = prediction_2[:, :, ::2, ::2]

        mask_1 = mask[:, :, ::2, ::2]
        mask_2 = mask_1[:, :, ::2, ::2]
        mask_3 = mask_2[:, :, ::2, ::2]

        gt_1 = gt[:, :, ::2, ::2]
        gt_2 = gt_1[:, :, ::2, ::2]
        gt_3 = gt_2[:, :, ::2, ::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    def LinearScaleInvarianceFramework(self, prediction, gt, mask, w_grad):

        assert (prediction.size(1) == gt.size(1))
        assert (prediction.size(1) == mask.size(1))

        w_data = 1.0
        # w_grad = 0.5
        gt_vec = gt[mask > 0.1]
        pred_vec = prediction[mask > 0.1]
        gt_vec = gt_vec.unsqueeze(1).float().cpu()
        pred_vec = pred_vec.unsqueeze(1).float().cpu()

        scale, _ = torch.gels(gt_vec.data, pred_vec.data)
        scale = scale[0, 0]

        # print("scale" , scale)
        # sys.exit()
        prediction_scaled = prediction * scale
        final_loss = w_data * self.L2Loss(prediction_scaled, mask, gt)

        prediction_1 = prediction_scaled[:, :, ::2, ::2]
        prediction_2 = prediction_1[:, :, ::2, ::2]
        prediction_3 = prediction_2[:, :, ::2, ::2]

        mask_1 = mask[:, :, ::2, ::2]
        mask_2 = mask_1[:, :, ::2, ::2]
        mask_3 = mask_2[:, :, ::2, ::2]

        gt_1 = gt[:, :, ::2, ::2]
        gt_2 = gt_1[:, :, ::2, ::2]
        gt_3 = gt_2[:, :, ::2, ::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_scaled, mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    def WeightedLinearScaleInvarianceFramework(self, prediction, gt, mask, w_grad):
        w_data = 1.0

        assert (prediction.size(1) == gt.size(1))
        assert (prediction.size(1) == mask.size(1))

        if torch.sum(mask.data) < 10:
            return 0

        # w_grad = 0.5
        gt_vec = gt[mask > 0.1]
        pred_vec = prediction[mask > 0.1]
        gt_vec = gt_vec.unsqueeze(1).float().cpu()
        pred_vec = pred_vec.unsqueeze(1).float().cpu()

        scale, _ = torch.gels(gt_vec.data, pred_vec.data)
        scale = scale[0, 0]

        prediction_scaled = prediction * scale

        ones_matrix = Variable(torch.zeros(gt.size(0), gt.size(1), gt.size(2), gt.size(3)) + 1, requires_grad=False)
        weight = torch.min(1 / gt, ones_matrix.float().cuda())
        weight_mask = torch.mul(weight, mask)

        final_loss = w_data * self.L2Loss(prediction_scaled, weight_mask, gt)

        prediction_1 = prediction_scaled[:, :, ::2, ::2]
        prediction_2 = prediction_1[:, :, ::2, ::2]
        prediction_3 = prediction_2[:, :, ::2, ::2]

        mask_1 = weight_mask[:, :, ::2, ::2]
        mask_2 = mask_1[:, :, ::2, ::2]
        mask_3 = mask_2[:, :, ::2, ::2]

        gt_1 = gt[:, :, ::2, ::2]
        gt_2 = gt_1[:, :, ::2, ::2]
        gt_3 = gt_2[:, :, ::2, ::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_scaled, weight_mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    def __call__(self, input_images, prediction_R, prediction_S, targets, data_set_name='IIW', epoch=None):

        if data_set_name == "IIW":
            # print("IIW Loss")
            num_images = prediction_R.size(0)
            # Albedo smoothness term
            # rs_loss =  self.w_rs_dense * self.BilateralRefSmoothnessLoss(prediction_R, targets, 'R', 5)
            # multi-scale smoothness term
            prediction_R_1 = prediction_R[:, :, ::2, ::2]
            prediction_R_2 = prediction_R_1[:, :, ::2, ::2]
            prediction_R_3 = prediction_R_2[:, :, ::2, ::2]

            rs_loss = self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R, targets, 0)
            rs_loss = rs_loss + 0.5 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_1, targets, 1)
            rs_loss = rs_loss + 0.3333 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_2, targets, 2)
            rs_loss = rs_loss + 0.25 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_3, targets, 3)

            #if opt.is_debug:
                #pdb.set_trace()
                #print([k for k in targets.keys()])
                #print(targets['S' + 'B_list'])
            #    pass

            # # Lighting smoothness Loss
            if self.w_ss_dense > 0:
                ss_loss = self.w_ss_dense * self.BilateralRefSmoothnessLoss(prediction_S, targets, 'S', 2)
            else:
                ss_loss = self.Tensor([0.0])

            # # Reconstruction Loss
            if self.w_reconstr_real > 0:
                reconstr_loss = self.w_reconstr_real * self.IIWReconstLoss(torch.exp(prediction_R), \
                                                                           torch.exp(prediction_S), targets)
            else:
                reconstr_loss = self.Tensor([0.0])

            # IIW Loss
            total_iiw_loss = Variable(torch.cuda.FloatTensor(1))
            total_iiw_loss[0] = 0

            for i in range(0, num_images):
                # judgements = json.load(open(targets["judgements_path"][i]))
                # total_iiw_loss += self.w_IIW * self.Ranking_Loss(prediction_R[i,:,:,:], judgements, random_filp)
                judgements_eq = targets["eq_mat"][i]
                judgements_ineq = targets["ineq_mat"][i]
                random_filp = targets["random_filp"][i]

                total_iiw_loss += self.w_IIW * self.BatchRankingLoss(prediction_R[i, :, :, :], judgements_eq,
                                                                     judgements_ineq, random_filp)

            total_iiw_loss = (total_iiw_loss) / num_images

            # print("reconstr_loss ", reconstr_loss.data[0])
            # print("rs_loss ", rs_loss.data[0])
            # print("ss_loss ", ss_loss.data[0])
            # print("total_iiw_loss ", total_iiw_loss.data[0])

            total_loss = total_iiw_loss + reconstr_loss + rs_loss + ss_loss

        else:
            print("NORMAL Loss")
            raise NotImplementedError

        self.total_loss = total_loss

        # return total_loss.data[0]
        return total_loss.data[0], rs_loss.data[0], ss_loss.data[0], total_iiw_loss.data[0]

    def compute_whdr(self, reflectance, judgements, delta=0.1):
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = reflectance.shape[0:2]

        error_sum = 0.0
        error_equal_sum = 0.0
        error_inequal_sum = 0.0

        weight_sum = 0.0
        weight_equal_sum = 0.0
        weight_inequal_sum = 0.0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            # convert to grayscale and threshold
            l1 = max(1e-10, np.mean(reflectance[
                                    int(point1['y'] * rows), int(point1['x'] * cols), ...]))
            l2 = max(1e-10, np.mean(reflectance[
                                    int(point2['y'] * rows), int(point2['x'] * cols), ...]))

            # # convert algorithm value to the same units as human judgements
            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'

            if darker == 'E':
                if darker != alg_darker:
                    error_equal_sum += weight

                weight_equal_sum += weight
            else:
                if darker != alg_darker:
                    error_inequal_sum += weight

                weight_inequal_sum += weight

            if darker != alg_darker:
                error_sum += weight

            weight_sum += weight

        if weight_sum:
            return (error_sum / weight_sum), error_equal_sum / (weight_equal_sum + 1e-10), error_inequal_sum / (
                        weight_inequal_sum + 1e-10)
        else:
            return None

    def evaluate_WHDR(self, prediction_R, targets):
        # num_images = prediction_S.size(0) # must be even number
        total_whdr = float(0)
        total_whdr_eq = float(0)
        total_whdr_ineq = float(0)

        count = float(0)

        for i in range(0, prediction_R.size(0)):
            prediction_R_np = prediction_R.data[i, :, :, :].cpu().numpy()
            prediction_R_np = np.transpose(np.exp(prediction_R_np * 0.4545), (1, 2, 0))

            # o_h = targets['oringinal_shape'][0].numpy()
            # o_w = targets['oringinal_shape'][1].numpy()

            # prediction_R_srgb_np = prediction_R_srgb.data[i,:,:,:].cpu().numpy()
            # prediction_R_srgb_np = np.transpose(prediction_R_srgb_np, (1,2,0))

            o_h = targets['oringinal_shape'][0].numpy()
            o_w = targets['oringinal_shape'][1].numpy()
            # resize to original resolution
            prediction_R_np = resize(prediction_R_np, (o_h[i], o_w[i]), order=1, preserve_range=True)

            # print(targets["judgements_path"][i])
            # load Json judgement
            judgements = json.load(open(targets["judgements_path"][i]))
            whdr, whdr_eq, whdr_ineq = self.compute_whdr(prediction_R_np, judgements, 0.1)

            total_whdr += whdr
            total_whdr_eq += whdr_eq
            total_whdr_ineq += whdr_ineq
            count += 1.

        return total_whdr, total_whdr_eq, total_whdr_ineq, count

    def evaluate_RC_loss(self, prediction_n, targets):

        normal_norm = torch.sqrt(torch.sum(torch.pow(prediction_n, 2), 1))
        normal_norm = normal_norm.unsqueeze(1).repeat(1, 3, 1, 1)
        prediction_n = torch.div(prediction_n, normal_norm)

        # mask_0 = Variable(targets['mask'].cuda(), requires_grad = False)
        # n_gt_0 = Variable(targets['normal'].cuda(), requires_grad = False)

        total_loss = self.AngleLoss(prediction_n, targets)

        return total_loss.data[0]

    def evaluate_L0_loss(self, prediction_R, targets):
        # num_images = prediction_S.size(0) # must be even number
        total_whdr = float(0)
        count = float(0)

        for i in range(0, 1):
            prediction_R_np = prediction_R
            # prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
            # prediction_R_np = np.transpose(prediction_R_np, (1,2,0))

            # load Json judgement
            judgements = json.load(open(targets["judgements_path"][i]))
            whdr = self.compute_whdr(prediction_R_np, judgements, 0.1)

            total_whdr += whdr
            count += 1

        return total_whdr, count

    def get_loss_var(self):
        return self.total_loss


##################################################################################
# Test codes
##################################################################################

def test_encoder():
    from configs.intrinsic import opt

    # enc = Vgg19Encoder(input_dim=opt.model.gen.input_dim, out_feature_dim=opt.model.gen.feature_dim,
    #                    pretrained=opt.model.gen.vgg_pretrained).cuda()
    enc = Vgg11EncoderMS(input_dim=opt.model.gen.input_dim, pretrained=opt.model.gen.vgg_pretrained).cuda()

    for i in range(19):
        input_data = torch.rand([opt.data.batch_size, opt.data.input_dim_a, opt.data.new_size, opt.data.new_size])
        input_data = Variable(input_data).cuda()

        out = enc(input_data)

        print(out['out'].shape, out['low'].shape)


def test_decoder():
    from configs.intrinsic import opt

    dec = Decoder(opt.model.gen.feature_dim, opt.model.gen.dim, opt.model.gen.output_dim_r,
                  opt.model.gen.n_layers, opt.model.gen.pad_type,
                  opt.model.gen.activ, opt.model.gen.norm).cuda()
    print(dec)

    input_data = torch.rand([opt.data.batch_size, opt.model.gen.feature_dim, opt.data.new_size, opt.data.new_size])
    input_data = Variable(input_data).cuda()

    output = dec(input_data)

    print(output.shape)


def test_gen():
    from configs.intrinsic import opt
    import numpy as np

    gen = TwoWayGenMS(opt.model.gen).cuda()

    for i in range(20):
        input_data = torch.rand([opt.data.batch_size, opt.model.gen.input_dim, opt.data.new_size, opt.data.new_size])
        input_data = Variable(input_data).cuda()

        output1, output2 = gen(input_data)[:2]
        output1 = output1.cpu().detach().numpy()
        output2 = output2.cpu().detach().numpy()

        print(np.mean(output1), np.std(output1), np.max(output1), np.min(output1))
        print(np.mean(output2), np.std(output2), np.max(output2), np.min(output2))

        print(output1.shape)
        print(output2.shape)
        print('-' * 40)


def test_diver():
    data1 = torch.rand([3, 32, 32, 32])
    data2 = torch.rand([3, 32, 32, 32])

    kl = KLDivergence()
    js = JSDivergence()

    out1 = kl(data1, data2)
    out2 = js(data1, data2)

    print(out1)
    print(out2)


def test_distance_loss():
    data1 = torch.Tensor([[1, 2], [2, 3]])
    data2 = torch.Tensor([[3, -9], [5, 6]])
    data1 = torch.rand([2, 3, 12, 12])
    data2 = torch.rand([2, 3, 12, 12])

    # cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    cos = DistanceLoss()

    out = cos(data1, data2)

    print(torch.mean(out))


def test_gradient_img():
    from torch.utils.data import DataLoader
    from utils import tensor2img, show_image
    from configs.intrinsic import opt
    import cv2
    import data

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
