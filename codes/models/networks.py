import logging
import torch
import torch.nn as nn
import models.modules.discriminators as D_arch
import models.modules.generators as G_arch
import models.modules.auxiliary_features as aux_arch
import torch.nn.init as init
import math
logger = logging.getLogger('base')

####################
# define network
####################

# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'sr_resnet':  # SRResNet
        netG = G_arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu')
    if which_model == 'vanilla':  # Vanilla
        netG = G_arch.VanillaNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
                nb=opt_net['nb'])
    elif which_model == 'sl_resnet':  # bottleneck spatial temporal res network 
        netG = G_arch.SLResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], inter_nc=opt_net['inter_nc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu')
    elif which_model == 'RRDB_net':  # RRDB
        netG = G_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'])
    elif which_model == 'fsrcnn':
        netG = G_arch.FSRCNN(scale_factor=int(opt_net['scale']), num_channels=opt_net['in_nc'], d=opt_net['nf'], s=opt_net['bk_nc'], m=opt_net['nb'])

    if opt['is_train']: 
        if which_model != 'fsrcnn': # fsrcnn has its own initialization
            initialize_weights(netG, scale=0.1)
        else:
            initialize_FSRCNN_weights(netG)

    return netG

# 8bit Quantization network
def define_Quant(model_fp32):
    return G_arch.QuantizedModel(model_fp32)

# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_64':
        netD = D_arch.Discriminator_VGG_64(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_64_SN':
        netD = D_arch.Discriminator_VGG_64_SN(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'])
    elif which_model == 'wgan_discriminator_vgg_64':
        netD = D_arch.WGAN_Discriminator_VGG_64(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], act_type=opt_net['act_type'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    initialize_weights(netD, scale=1)
    # if gpu_ids:
    #     netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        # conv54 34, conv44 25
        feature_layer = 34
    netF = aux_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF

####################
# initialization
####################

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_FSRCNN_weights(net_l):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.first_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in net.mid_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(net.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(net.last_part.bias.data)
