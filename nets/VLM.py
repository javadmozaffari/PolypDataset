# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.ops import SqueezeExcitation

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule
from .swinunet import PatchEmbed, BasicLayer
from .imgencoder import GroupMixFormer
# from .textencoder import TextEncoder
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)





class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MSAM(nn.Module):
    def __init__(self, ch1,ch2,ch3):
        super(MSAM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(128, 64, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(128, 64, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(64, 32, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(128, 32, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(64, 32, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(32, 16, 3, padding=1)

        self.conv_input1 = BasicConv2d(ch1, 128, 3, padding=1)
        self.conv_input2 = BasicConv2d(ch2, 64, 3, padding=1)
        self.conv_input3 = BasicConv2d(ch3, 32, 3, padding=1)
    
        
        self.upsample0_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample0_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample0_3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        # self.outMSAM = BasicConv2d(128, 64, 3, padding=1)

    def forward(self, x1, x2, x3):
        
        
        x1 = self.conv_input1(x1)
        x2 = self.conv_input2(x2)
        x3 = self.conv_input3(x3)
        
        x1_1 = self.conv_upsample1(self.upsample(x1))
        x1_2 = self.conv_upsample1(self.upsample(x1)) 
        x1_3 = x1_2* x2
        x1_3 = torch.cat((x1_3, x1_1), 1)
        x1_3 = self.conv_upsample2(x1_3)
        x1_3 = self.upsample(x1_3)
        x1_3 = self.conv_upsample3(x1_3)
        x1_4 = self.conv_upsample4(self.upsample4(x1))
        
        x2_1 = self.conv_upsample3(self.upsample(x2))

        x3_1 = x2_1*x1_4*x3
        x3_2 = torch.cat((x3_1, x1_3), 1)
        x3_2 = self.conv_upsample5(x3_2)
        x3_2 = self.conv_upsample6(x3_2)
        
        return x3_2



class GAM(nn.Module):
    def __init__(self, ch0, ch1,ch2,ch3):
        super(GAM, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.conv_upsample1 = BasicConv2d(128, 32, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(128, 64, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(64, 32, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(128, 32, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(64, 32, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(32, 16, 3, padding=1)

        self.conv_input1 = BasicConv2d(ch1, 128, 3, padding=1)
        self.conv_input2 = BasicConv2d(ch2, 64, 3, padding=1)
        self.conv_input3 = BasicConv2d(ch3, 32, 3, padding=1)
        

        
        self.conv_0_1 = BasicConv2d(ch0, 128, 3, padding=1)
        self.conv_0_2 = BasicConv2d(ch0, 64, 3, padding=1)
        self.conv_0_3 = BasicConv2d(ch0, 32, 3, padding=1)
        
        self.upsample0_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample0_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample0_3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        self.outmsam = BasicConv2d(128, 64, 3, padding=1)

    def forward(self, x0,  x1, x2, x3):
        
        x1 = self.conv_input1(x1)
        x2 = self.conv_input2(x2)
        x3 = self.conv_input3(x3)
        

        x0_1 = self.conv_0_1(self.upsample0_1(x0))
        x0_2 = self.conv_0_2(self.upsample0_2(x0))
        x0_3 = self.conv_0_3(self.upsample0_3(x0))
        


        x1 = (x0_1+x1)*x1
        x2 = (x0_2+x2)*x2
        x3 = (x0_3+x3)*x3

    
        x1_1 = self.conv_upsample1(self.upsample4(x1))
        x1_2 = self.conv_upsample2(self.upsample(x1)) 
        x1_3 = x1_2* x2
        x1_3 = self.upsample(x1_3)


        x2_1 = self.conv_upsample3(self.upsample(x2))
        x2_2 = x2_1*x3        
        
        x3_1 = x1_1*x3
        x3_2 = x3_1 + x2_2
        
        x_out = torch.cat((x3_2, x1_3, x3), 1)
        x_out = self.outmsam(x_out)
    
        return x_out        
        
        
        
class VLM(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.outc2 = nn.Conv2d(16, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))

        self.text_module4 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=4, in_chans=3, embed_dim=96,
            norm_layer=nn.LayerNorm)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        i_layer = 0    
        depths=[2, 2, 2, 2]
        num_heads=[3, 6, 12, 24]
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(96 * 2 ** i_layer),
                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                             patches_resolution[1] // (2 ** i_layer)),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=7,
                        mlp_ratio=4.,
                        qkv_bias=True, qk_scale=None,
                        drop=0., attn_drop=0.,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=nn.LayerNorm,
                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                        use_checkpoint=None)
            self.layers.append(layer)

        self.conv1 = ConvBatchNorm(80, 64)
        self.conv2 = ConvBatchNorm(160, 128)
        self.conv3 = ConvBatchNorm(320, 256)
        self.conv4 = ConvBatchNorm(320, 512)
        
        
        self.conv11 = ConvBatchNorm(64, 32)
        self.conv12 = ConvBatchNorm(32, 1)
        self.conv21 = ConvBatchNorm(64, 16)
        
        self.imgenc = GroupMixFormer()
        self.gam = GAM(512, 256, 128, 64)
        self.msam = MSAM(256, 128, 64)
        self.se = SqueezeExcitation(512, 128)



    def up_x4(self, x, s):
        H, W = self.patches_resolution
        B, L, C = x.shape
        x = x.view(B,int(s*H),int(s*W),-1)
        x = x.permute(0,3,1,2) 
        return x
        
    def forward(self, x, text):

        image_encoder = self.imgenc(x)
        image_block1 = image_encoder[0]
        image_block2 = image_encoder[1]
        image_block3 = image_encoder[2]
        image_block4 = image_encoder[3]

        x1 = self.conv1(image_block1)
        x1 = F.interpolate(input= x1, scale_factor =4)  

        x2 = self.conv2(image_block2)
        x2 = F.interpolate(input= x2, scale_factor =4)  

        x3 = self.conv3(image_block3)
        x3 = F.interpolate(input= x3, scale_factor =4)  

        x4 = self.conv4(image_block4)
        x4 = F.interpolate(input= x4, scale_factor =4)          
 
        text = torch.squeeze(text, dim=1)
        text_block4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        text_block3 = self.text_module3(text_block4.transpose(1, 2)).transpose(1, 2)
        text_block2 = self.text_module2(text_block3.transpose(1, 2)).transpose(1, 2)
        text_block1 = self.text_module1(text_block2.transpose(1, 2)).transpose(1, 2)


        fusion_block1 = self.downVit(x1, x1, text_block1)
        fusion_block2 = self.downVit1(x2, fusion_block1, text_block2)
        fusion_block3 = self.downVit2(x3, fusion_block2, text_block3)
        fusion_block4 = self.downVit3(x4, fusion_block3, text_block4)

        upfusion4 = self.upVit3(fusion_block4, fusion_block4, text_block4, True)
        upfusion3 = self.upVit2(fusion_block3, upfusion4, text_block3, True)
        upfusion2 = self.upVit1(fusion_block2, upfusion3, text_block2, True)
        upfusion1 = self.upVit(fusion_block1, upfusion2, text_block1, True)

        upfusion1 = self.reconstruct1(upfusion1) 
        upfusion2 = self.reconstruct2(upfusion2) 
        upfusion3 = self.reconstruct3(upfusion3)
        upfusion4 = self.reconstruct4(upfusion4)

        upfusion4 = self.se(upfusion4)
        out1 = self.gam(upfusion4, upfusion3, upfusion2,upfusion1)
        
        out1_2 = out1
        out1 = self.conv11(out1)
        out2 = self.msam(upfusion3, upfusion2, out1_2)

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(out1))
            logits2 = self.last_activation(self.outc2(out2))
        else:
            logits = self.outc(out1) 
            logits2 = self.outc2(out2)

        return logits, logits2