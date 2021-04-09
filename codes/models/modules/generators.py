import torch.nn as nn
import torch
import torch.nn.functional as F
from . import block as B

####################
#  SRResNet
####################
class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='batch', act_type='relu'):
        super(SRResNet, self).__init__()

        # feature extraction/ denoise
        fea_conv = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        resnet_blocks = [ResNetBlock(nf, nf, 3, act_type=act_type) for _ in range(nb)]
        lr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        # upsample 
        up = nn.Upsample(scale_factor=(upscale, 1., 1.,), mode='nearest')
        up_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        # output
        hr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        last_conv = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)

        if upscale == 1:# denoising
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                                    hr_conv, last_conv)
        else: # super-res
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                                    up, up_conv, hr_conv, last_conv)

    def forward(self, x):
        return self.net(x)


class QuantizedModel(nn.Module):
    # https://leimao.github.io/blog/PyTorch-Static-Quantization/
    def __init__(self, model_fp32):
        super(QuantizedModel, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


# WGAN paper  https://arxiv.org/abs/1708.00961 only for denoising
# Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance
# and Perceptual Loss
class VanillaNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(VanillaNet, self).__init__()
        layers = [nn.Conv3d(in_nc, nf, 3, 1, 1), nn.ReLU()]
        for _ in range(2, nb):
            layers.extend([nn.Conv3d(nf, nf, 3, 1, 1), nn.ReLU()])
        layers.extend([nn.Conv3d(nf, out_nc, 3, 1, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, dilation=1, bias=True, \
                 norm_type=None, act_type='relu', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = B.conv_block(in_nc, out_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        conv1 = B.conv_block(out_nc, out_nc, kernel_size, stride, dilation, bias, norm_type, None)
        self.res = B.sequential(conv0, conv1)
        self.res_scale = res_scale
        self.skip_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        res = self.skip_op.mul_scalar(self.res(x), self.res_scale)
        return self.skip_op.add(x, res)


####################
# FSRCNN
####################

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=1, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv3d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv3d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv3d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose3d(d, num_channels, kernel_size=(9, 5, 5), stride=(scale_factor, 1, 1), padding=(4, 2, 2),
                                            output_padding=(scale_factor-1, 0, 0))

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        return self.last_part(x)


####################
# SLResNet
####################

class SLResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, inter_nc, upscale=1, norm_type='batch', act_type='relu'):
        super(SLResNet, self).__init__()

        # feature extraction/ denoise
        # fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        fea_conv = nn.Conv3d(in_nc, nf, kernel_size=5, padding=5//2, bias=True)
        resnet_blocks = [SL_A_ResNetBlock(nf, nf, inter_nc, 3, act_type=act_type) for _ in range(nb)]
        lr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        # upsample 
        up = nn.Upsample(scale_factor=(upscale, 1., 1.,), mode='nearest')
        up_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        # output
        hr_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)

        if upscale == 1:# denoising
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                    hr_conv, conv_last)
        else: #super-res
            self.net = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, lr_conv)),
                                up, up_conv, hr_conv, conv_last)

    def forward(self, x):
        return self.net(x)

class SL_A_ResNetBlock(nn.Module):

    def __init__(self, in_nc, out_nc, inter_nc, kernel_size=3, stride=1, dilation=1, bias=True, \
                 norm_type=None, act_type='relu'):
        super(SL_A_ResNetBlock, self).__init__()

        spa_conv = B.s_conv_block(in_nc, inter_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        temp_conv = B.t_conv_block(inter_nc, out_nc, kernel_size, stride, dilation, bias, norm_type, None)
        self.res = B.sequential(spa_conv, temp_conv)
        self.skip_op = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_op.add(x, self.res(x))

class SL_B_ResNetBlock(nn.Module):

    def __init__(self, in_nc, out_nc, bk_nc, inter_nc, kernel_size=3, stride=1, dilation=1, bias=True, \
                 norm_type=None, act_type='relu'):
        super(SL_B_ResNetBlock, self).__init__()

        self.spa_conv = B.s_conv_block(in_nc, inter_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        self.temp_conv = B.t_conv_block(in_nc, inter_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        self.conv1 = B.conv_block(inter_nc, out_nc, 1, stride, dilation, bias, norm_type, None)

        self.is_not_out_nc_equal_to_inter_nc = out_nc != inter_nc

    def forward(self, x):
        out = self.spa_conv(x) + self.temp_conv(x) # parallel spatial temporal
        if self.is_not_out_nc_equal_to_inter_nc:
            out = self.conv1(out) # 1x1x1 conv when out_nc != inter_nc
        return x + out

class SL_C_ResNetBlock(nn.Module):

    def __init__(self, in_nc, out_nc, bk_nc, inter_nc, kernel_size=3, stride=1, dilation=1, bias=True, \
                 norm_type=None, act_type='relu'):
        super(SL_C_ResNetBlock, self).__init__()

        self.spa_conv = B.s_conv_block(in_nc, inter_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        self.temp_conv = B.t_conv_block(inter_nc, inter_nc, kernel_size, stride, dilation, bias, norm_type, act_type)
        self.conv1 = B.conv_block(inter_nc, out_nc, 1, stride, dilation, bias, norm_type, None)

        self.is_not_out_nc_equal_to_inter_nc = out_nc != inter_nc

    def forward(self, x):
        out = self.spa_conv(x) 
        out = out + self.temp_conv(out) # fuse spatial & temporal
        if self.is_not_out_nc_equal_to_inter_nc:
            out = self.conv1(out) # 1x1x1 conv when out_nc != inter_nc
        return x + out


####################
# RRDB
####################

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=1, norm_type=None):
        super(RRDBNet, self).__init__()
        self.upscale = upscale

        # feature extraction/ denoise
        fea_conv = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        rb_blocks = [RRDB(nf, gc) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None)
        self.out_feat = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
        # upsample x2
        self.upconv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # output
        self.HR_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.out_feat(x)
        if self.upscale == 2:
            x = self.lrelu(self.upconv(F.interpolate(x, scale_factor=(2., 1., 1.), mode='nearest')))
        # out = self.conv_last(x)
        return self.conv_last(self.lrelu(self.HR_conv(x)))


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nf, gc, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = B.conv_block(nf, gc, kernel_size, stride, bias=bias, \
            norm_type=norm_type, act_type=act_type)
        self.conv2 = B.conv_block(nf+gc, gc, kernel_size, stride, bias=bias, \
            norm_type=norm_type, act_type=act_type)
        self.conv3 = B.conv_block(nf+2*gc, gc, kernel_size, stride, bias=bias, \
            norm_type=norm_type, act_type=act_type)
        self.conv4 = B.conv_block(nf+3*gc, gc, kernel_size, stride, bias=bias, \
            norm_type=norm_type, act_type=act_type)
        self.conv5 = B.conv_block(nf+4*gc, nf, 3, stride, bias=bias, \
            norm_type=norm_type, act_type=None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
