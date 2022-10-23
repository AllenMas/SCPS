import torch
import torch.nn as nn
import torch.nn.functional as F

def activation(afunc='LReLU'):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=True)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        raise Exception('Unknown activation function')

def conv_layer(batchNorm, cin, cout, k=3, stride=1, pad=-1, afunc='LReLU'):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    mList = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True)]
    if batchNorm:
        print('=> convolutional layer with batchnorm')
        mList.append(nn.BatchNorm2d(cout))
    mList.append(activation(afunc))
    return nn.Sequential(*mList)

# Model
class Light_Model(nn.Module):
    def __init__(self, num_rays, light_init, requires_grad=True):
        super(Light_Model, self).__init__()

        light_direction_xy = light_init[0][:, :-1].clone().detach()  # (144,2)
        light_direction_z = light_init[0][:, -1:].clone().detach()  # (144,1)
        light_intensity = light_init[1].mean(dim=-1, keepdims=True).clone().detach()  # (144,1)

        self.light_direction_xy = nn.Parameter(light_direction_xy.float(), requires_grad=requires_grad)
        self.light_direction_z = nn.Parameter(light_direction_z.float(), requires_grad=requires_grad)
        self.light_intensity = nn.Parameter(light_intensity.float(), requires_grad=requires_grad)

        self.num_rays = num_rays

    def forward(self, idx):
        num_rays = self.num_rays
        out_ld = torch.cat([self.light_direction_xy[idx], -torch.abs(self.light_direction_z[idx])], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)[:, None, :]  # (96, 1, 3)

        out_ld = out_ld.repeat(1, num_rays, 1)
        out_ld = out_ld.view(-1, 3)  # (96*num_rays, 3)

        out_li = torch.abs(self.light_intensity[idx])[:, None, :]  # (96, 1, 1)
        out_li = out_li.repeat(1, num_rays, 3)
        out_li = out_li.view(-1, 3)  # (96*num_rays, 3)
        return out_ld, out_li

    def get_light_from_idx(self, idx):
        out_ld_r, out_li_r = self.forward(idx)
        return out_ld_r, out_li_r

    def get_all_lights(self):
        with torch.no_grad():
            light_direction_xy = self.light_direction_xy
            light_direction_z = -torch.abs(self.light_direction_z)
            light_intensity = torch.abs(self.light_intensity).repeat(1, 3)

            out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
            out_ld = F.normalize(out_ld, p=2, dim=-1)
            return out_ld, light_intensity  # (n, 3), (n, 3)


class Light_Model_CNN(nn.Module):
    def __init__(
            self,
            num_layers=3,
            hidden_size=64,
            output_ch=4,
            batchNorm=False
    ):
        super(Light_Model_CNN, self).__init__()
        self.conv1 = conv_layer(batchNorm, 4, 64,  k=3, stride=2, pad=1, afunc='LReLU')
        self.conv2 = conv_layer(batchNorm, 64, 128,  k=3, stride=2, pad=1)
        self.conv3 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv4 = conv_layer(batchNorm, 128, 128,  k=3, stride=2, pad=1)
        self.conv5 = conv_layer(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.conv6 = conv_layer(batchNorm, 128, 256,  k=3, stride=2, pad=1)
        self.conv7 = conv_layer(batchNorm, 256, 256,  k=3, stride=1, pad=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = torch.nn.functional.relu
        self.dir_linears = nn.ModuleList(
            [nn.Linear(256, hidden_size)] + [nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 1)])
        self.output_linear = nn.Linear(hidden_size, output_ch)

    def forward(self, inputs):
        x = inputs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        for i, l in enumerate(self.dir_linears):
            out = self.dir_linears[i](out)
            out = self.relu(out)
        outputs = self.output_linear(out)

        light_direction_xy = outputs[:, :2]
        light_direction_z = -torch.abs(outputs[:, 2:3])-0.1
        light_intensity = torch.abs(outputs[:, 3:])

        out_ld = torch.cat([light_direction_xy, light_direction_z], dim=-1)
        out_ld = F.normalize(out_ld, p=2, dim=-1)  # (96, 3)
        out_li = light_intensity  # (96, 1)

        outputs = {}
        outputs['dirs'] = out_ld
        outputs['ints'] = out_li
        return outputs

    def set_images(self, num_rays, images, device):
        self.num_rays = num_rays
        self.images = images
        self.device = device
        return

    def get_all_lights(self):
        images = iter(self.images.split(len(self.images), dim=0))

        inputs = next(images)
        outputs = self.forward(inputs.to(self.device))
        out_ld, out_li = outputs['dirs'], outputs['ints']

        for inputs in images:
            outputs = self.forward(inputs.to(self.device))
            out_ld = torch.cat([out_ld, outputs['dirs']], dim=0)
            out_li = torch.cat([out_li, outputs['ints']], dim=0)

        return out_ld, out_li.repeat(1, 3)