import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn as nn
from  .selectdevice import sel
device=sel()
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn as nn

activations = {'ReLU': nn.ReLU,
               'LeakyReLU': nn.LeakyReLU,
               'ReLU6': nn.ReLU6,
               'SELU': nn.SELU,
               'ELU': nn.ELU,
               None: nn.Identity
               }


class LA_Block(nn.Module):
    def __init__(self, channel, x, y, reduction=16, ratio=4):
        super(LA_Block, self).__init__()
        self.x = x
        self.y = y
        self.channel = channel
        self.up = nn.Upsample(scale_factor=2).to(device)
        self.conv = Conv(c1=channel // reduction, c2=channel, k=1).to(device)
        self.avg_pool_x = nn.AdaptiveAvgPool2d((x * 2, 1)).to(device)
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, y * 2)).to(device)
        self.relu = nn.ReLU().to(device)
        self.bn = nn.BatchNorm2d(channel // reduction).to(device)
        self.conv1X1 = Conv(c1=channel, c2=channel // reduction, k=1).to(device)
        self.sigmoid_x = nn.Sigmoid().to(device)
        self.sigmoid_y = nn.Sigmoid().to(device)
        self.conv2 = Conv(c1=channel, c2=channel, k=1, s=2).to(device)

    def forward(self, x):
        tar = x
        up = self.up(x)
        x, y = self.avg_pool_x(up), self.avg_pool_y(up)
        x, y = self.conv1X1(x), self.conv1X1(y)
        x_cat_conv_bn_relu = self.relu(self.bn(torch.add(x, y)))
        x_sig = self.sigmoid_x(x_cat_conv_bn_relu)
        x_conv = self.conv2(self.conv(x_sig))
        out = tar * x_conv.expand_as(tar)
        return out


def act_layers(name):
    assert name in activations.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    else:
        return activations[name](inplace=True)


model_urls = {
    'shufflenetv2_0.5x': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_1.0x': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_1.5x': None,
    'shufflenetv2_2.0x': None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g)
        )

    def forward(self, x):
        return self.convs(x)


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation='ReLU'):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 model_size='1.5x',
                 out_stages=(2, 3, 4),
                 with_last_conv=False,
                 kernal_size=3,
                 activation='ReLU', train=True):
        super(ShuffleNetV2, self).__init__()
    

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == '0.5x':
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2, activation=activation)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1, activation=activation))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            self.conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module('conv5', self.conv5)
        if train:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print('init weights...')
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if pretrain:
            url = model_urls['shufflenetv2_{}'.format(self.model_size)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                self.load_state_dict(pretrained_state_dict, strict=False)


class SPP(nn.Module):
    """Spatial Pyramid Pooling
    """

    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x
class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
    def forward(self,*x):
        p3,p4,p5=x[0],x[1],x[2]#提取出来特征层  8 16 32
        p4=torch.cat(
            (F.interpolate(p5,scale_factor=2),p4),dim=1)
        p3=torch.cat((F.interpolate(p4,scale_factor=2),p3),dim=1)
        return  [p3,p4,p5]#8 16  32
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                                                mode=self.mode, align_corners=self.align_corner)

class PAN(nn.Module):
    def __init__(self, classes):
        super(PAN, self).__init__()

        # head
        self.head_conv_0 = Conv(464*4, 232, k=1)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(464, 232, n=1, shortcut=False)

        # P3/8-small
        self.head_conv_1 = Conv(232, 116, k=1)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(232, 116, n=1, shortcut=False)

        # P4/16-medium
        self.head_conv_2 = Conv(116, 116, k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(116+116, 232, n=1, shortcut=False)

        # P8/32-large
        self.head_conv_3 = Conv(232, 232, k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(232+232, 464, n=1, shortcut=False)

        # det conv
        self.head_det_1 = nn.Conv2d(116, 3* (1 +classes + 4), 1)
        self.head_det_2 = nn.Conv2d(232, 3 * (1 + classes + 4), 1)
        self.head_det_3 = nn.Conv2d(464, 3 * (1 + classes+ 4), 1)
        self.spp=SPP()


    def forward(self, *x):
        c3, c4, c5 = (i for i in x)
        c5=self.spp(c5)
        """
        c5后面增加一个
        """
        # FPN + PAN
        # head
        """
        使用LA注意力机制 使分类特征更加显著
        """
        la_model = LA_Block(channel=232, x=c4.size()[-1], y=c4.size()[-1])
        c4 = la_model(c4)

        c6 = self.head_conv_0(c5)

        c7 = self.head_upsample_0(c6)  # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)  # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        #c14 116 52 52
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        # det
        pred_s = self.head_det_1(c13)
        pred_m = self.head_det_2(c16)
        pred_l = self.head_det_3(c19)
        preds = [pred_s, pred_m, pred_l]
        return preds

#model = ShuffleNetV2(model_size='1.0x', )
if __name__ == "__main__":
    model = ShuffleNetV2(model_size='1.0x', ).to(device)

    test_data = torch.rand(5, 3, 608, 608).to(device)
    # torch.Size([5, 116, 52, 52])
    # torch.Size([5, 232, 26, 26])
    # torch.Size([5, 464, 13, 13])
    pan=PAN(classes=20).to(device)
    test_outputs = model(test_data)
    _=pan(*test_outputs)
    for i in _:
        print(i.size())


