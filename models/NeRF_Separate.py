import torch
import torch.nn.functional as F
import torch.nn as nn
from .position_encoder import positional_encoding
from .MultiHashEncoding import HashEmbedder


class NormalConverter(nn.Module):
    def __init__(self, xy, mask, K):
        super(NormalConverter, self).__init__()
        self.index = torch.where(mask > 0.5)
        self.shape = mask.shape
        self.register_buffer("xy", xy)
        self.register_buffer("invK_T", torch.linalg.inv(K).permute(1, 0))
        self.register_buffer("weights", self.__weights_init__())  # w(3, 27, 3, 3), (b,3,h,w) -> (b,27,h,w)

    def __weights_init__(self):
        weights = torch.zeros((27, 3, 3, 3))
        weight = torch.tensor([[8 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float, requires_grad=False)
        weights[0, 0, :, :] = weight
        weights[1, 1, :, :] = weight
        weights[2, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, 8 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[3, 0, :, :] = weight
        weights[4, 1, :, :] = weight
        weights[5, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, 8 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[6, 0, :, :] = weight
        weights[7, 1, :, :] = weight
        weights[8, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, 8 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[9, 0, :, :] = weight
        weights[10, 1, :, :] = weight
        weights[11, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, 8 / 9]], dtype=torch.float32, requires_grad=False)
        weights[12, 0, :, :] = weight
        weights[13, 1, :, :] = weight
        weights[14, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, 8 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[15, 0, :, :] = weight
        weights[16, 1, :, :] = weight
        weights[17, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9],
                               [8 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[18, 0, :, :] = weight
        weights[19, 1, :, :] = weight
        weights[20, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [8 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[21, 0, :, :] = weight
        weights[22, 1, :, :] = weight
        weights[23, 2, :, :] = weight

        weight = torch.tensor([[-1 / 9, -1 / 9, -1 / 9],
                               [-1 / 9, 8 / 9, -1 / 9],
                               [-1 / 9, -1 / 9, -1 / 9]], dtype=torch.float32, requires_grad=False)
        weights[24, 0, :, :] = weight
        weights[25, 1, :, :] = weight
        weights[26, 2, :, :] = weight

        return weights

    def forward(self, depth):
        xyz = torch.matmul(torch.cat([self.xy * depth, depth], dim=-1), self.invK_T)  # (p, 3)

        cloudmap = torch.zeros([*self.shape, 3], dtype=torch.float32, device=depth.device)
        cloudmap[self.index] = xyz  # (h, w, 3)
        cloudmap = cloudmap.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, h, w)

        Y = F.conv2d(cloudmap, self.weights, stride=1, padding=1).squeeze(0).permute(1, 2, 0)  # (h, w, 27)
        Y = Y.reshape([*self.shape, 9, 3])  # (h, w, 9, 3)
        Y = Y[self.index]  # (p, 9, 3)
        Y = Y + torch.randn_like(Y, device=Y.device) * 1e-6  # (p, 9, 3)
        u, s, vt = torch.linalg.svd(Y)  # (p, 3, 3)
        normal = vt[:, 2]  # (p, 3)

        normal = normal * torch.where(normal[:, [2]] > 0, -1.0, 1.0)
        normal = F.normalize(normal, p=2, dim=-1)  # (p, 3)

        return normal


class NeRF(nn.Module):
    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=4,
            normalized_xy=None,
            mean_var=None,
            xy=None,
            mask=None,
            K=None
    ):
        super(NeRF, self).__init__()

        self.dim_ixiy = 40
        self.skip_connect_every = skip_connect_every

        self.converter = NormalConverter(xy, mask, K)
        self.relu = F.relu

        ''' encoding '''
        # encoded_xy = positional_encoding(normalized_xy, num_encoding_functions=10)
        # encoded_xy = torch.cat([encoded_xy, mean_var], dim=-1)
        # self.register_buffer("encoded_xy", encoded_xy)
        self.embedder = HashEmbedder(n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16,
                                     finest_resolution=mask.shape[0])
        self.register_buffer("normalized_xy", normalized_xy)
        self.register_buffer("scaled_xy", xy / torch.tensor([[mask.shape[1], mask.shape[0]]], dtype=torch.float32))
        self.register_buffer("mean_var", mean_var)

        ''' Base Feature Extractor '''
        self.flayers = torch.nn.ModuleList()
        self.flayers.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.flayers.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.flayers.append(torch.nn.Linear(hidden_size, hidden_size))
        ''' depth predictor '''
        self.fc_depth = torch.nn.Linear(hidden_size, 1)

        ''' Spec Diff Feature Extractor '''
        self.slayers = torch.nn.ModuleList()
        self.slayers.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
        self.slayers.append(torch.nn.Linear(hidden_size, hidden_size // 2))
        for i in range(3):
            self.slayers.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))

        ''' Spec Diff predictor '''
        self.fc_diff = torch.nn.Linear(hidden_size // 2, 3)
        self.fc_spec_coeff = torch.nn.Linear(hidden_size // 2, 36)

    def forward(self):
        '''
        @param
            input_xy: (p, 48), [x, y, x*si, y*si,..., mu_r, mu_g, mu_b, var_r, var_g, var_b]
            xy: (p, 2)
        Returns:
            depth: depth > 0, (p, 1)
            normal: nz < 0, shape(p, 3)
            diffuse: (p, 3)
            spec_coeff
        '''
        ##### Compute depth #####
        # x = self.encoded_xy
        encoded_xy = torch.cat([self.normalized_xy, self.embedder(self.scaled_xy), self.mean_var], dim=-1)
        x = encoded_xy
        for i in range(len(self.flayers)):
            if i == self.skip_connect_every:
                x = self.flayers[i](torch.cat((encoded_xy, x), -1))
            else:
                x = self.flayers[i](x)
            x = self.relu(x)
        depth = torch.abs(self.fc_depth(x)) + 1e-6
        normal = self.converter(depth)

        ##### Compute diff spec ######
        x = self.slayers[0](torch.cat((encoded_xy, x), -1))
        for i in range(1, len(self.slayers)):
            x = self.slayers[i](x)
            x = self.relu(x)
        diff = torch.abs(self.fc_diff(x))
        spec_coeff = self.fc_spec_coeff(x)

        return depth, normal, diff, spec_coeff


class NeRFSmall(nn.Module):
    def __init__(
            self,
            num_layers=2,
            hidden_size=64,
            normalized_xy=None,
            mean_var=None,
            xy=None,
            mask=None,
            K=None
    ):
        super(NeRFSmall, self).__init__()

        self.dim_ixiy = 40
        self.converter = NormalConverter(xy, mask, K)
        self.relu = F.relu

        ''' encoding '''
        self.embedder = HashEmbedder(n_levels=16, n_features_per_level=2, log2_hashmap_size=19,
                                     base_resolution=16, finest_resolution=mask.shape[0])
        self.register_buffer("normalized_xy", normalized_xy)
        self.register_buffer("scaled_xy", xy / torch.tensor([[mask.shape[1], mask.shape[0]]], dtype=torch.float32))
        self.register_buffer("mean_var", mean_var)

        ''' depth net '''
        self.dlayers = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = self.dim_ixiy
            else:
                in_dim = hidden_size

            if i == num_layers - 1:
                out_dim = hidden_size + 1
            else:
                out_dim = hidden_size
            self.dlayers.append(torch.nn.Linear(in_dim, out_dim))

        ''' Spec Diff '''
        self.slayers = torch.nn.ModuleList()
        for i in range(num_layers+1):
            if i == 0:
                in_dim = self.dim_ixiy + hidden_size
            else:
                in_dim = hidden_size

            if i == num_layers:
                out_dim = 39
            else:
                out_dim = hidden_size
            self.slayers.append(torch.nn.Linear(in_dim, out_dim))

    def forward(self):
        '''
        @param
            input_xy: (p, 48), [x, y, x*si, y*si,..., mu_r, mu_g, mu_b, var_r, var_g, var_b]
            xy: (p, 2)
        Returns:
            depth: depth > 0, (p, 1)
            normal: nz < 0, shape(p, 3)
            diffuse: (p, 3)
            spec_coeff
        '''
        ##### Compute depth #####
        encoded_xy = torch.cat([self.normalized_xy, self.embedder(self.scaled_xy), self.mean_var], dim=-1)
        x = encoded_xy
        for i in range(len(self.dlayers)):
            x = self.dlayers[i](x)
            if i != len(self.dlayers) - 1:
                x = self.relu(x)
        depth = torch.abs(x[:, [0]]) + 1e-6
        feat = self.relu(x[:, 1:])

        normal = self.converter(depth)

        ##### Compute diff spec #####
        x = torch.cat((encoded_xy, feat), -1)
        for i in range(len(self.slayers)):
            x = self.slayers[i](x)
            if i != len(self.slayers) - 1:
                x = self.relu(x)
        diff = torch.abs(x[:, 0:3])
        spec_coeff = x[:, 3:]

        return depth, normal, diff, spec_coeff
