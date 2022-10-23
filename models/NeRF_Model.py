import torch
import torch.nn.functional as F


class NeRFModel_Separate(torch.nn.Module):
    def __init__(
            self,
            num_layers=8,
            hidden_size=256,
            skip_connect_every=3,
            num_encoding_fn_input1=10,
            num_encoding_fn_input2=0,
            include_input_input1=2,  # denote images coordinates (ix, iy)
            include_input_input2=0,  # denote lighting direcions (lx, ly, lz)
            output_ch=1,
            gray_scale=False,
            mask=None,
    ):
        super(NeRFModel_Separate, self).__init__()
        self.dim_ldir = include_input_input2 * (1 + 2 * num_encoding_fn_input2)
        self.dim_ixiy = include_input_input1 * (1 + 2 * num_encoding_fn_input1) + self.dim_ldir
        self.dim_ldir = 0
        self.skip_connect_every = skip_connect_every + 1

        ##### Layers for Material Map #####
        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))
        ###################################

        ##### Layers for Normal Map #####
        self.layers_xyz_normal = torch.nn.ModuleList()
        self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_xyz_normal.append(torch.nn.Linear(self.dim_ixiy + hidden_size, hidden_size))
            else:
                self.layers_xyz_normal.append(torch.nn.Linear(hidden_size, hidden_size))
        ###################################
        self.relu = torch.nn.functional.relu
        self.register_buffer('mask', mask)
        self.idxp = torch.where(self.mask > 0.5)

        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(hidden_size + self.dim_ldir, hidden_size // 2))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        self.fc_spec_coeff = torch.nn.Linear(hidden_size // 2, output_ch)

        self.fc_diff = torch.nn.Linear(hidden_size // 2, 1 if gray_scale else 3)

        self.fc_normal_xy = torch.nn.Linear(hidden_size, 2)
        self.fc_normal_z = torch.nn.Linear(hidden_size, 1)

    def forward(self, input):
        '''
        @param
            input: shape(p, 48), 48=2+2*20+6,
                   e.g [x, y, x*si, y*si,..., mu_r, mu_g, mu_b, var_r, var_g, var_b]

        @returns
            normal: shape(p, 3)
            diff: shape(p, 3)
            spec_coeff: shape(p, 48)
        '''
        xyz = input[..., : self.dim_ixiy]

        ##### Compute Normal Map #####
        x = xyz
        for i in range(len(self.layers_xyz_normal)):
            if i == self.skip_connect_every:
                x = self.layers_xyz_normal[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz_normal[i](x)
            x = self.relu(x)
        normal_xy = self.fc_normal_xy(x)
        normal_z = -torch.abs(self.fc_normal_z(x))  # n_z is always facing camera
        normal = torch.cat([normal_xy, normal_z], dim=-1)
        normal = F.normalize(normal, p=2, dim=-1)
        ###################################

        ##### Compute Materaial Map #####
        x = xyz
        for i in range(len(self.layers_xyz)):
            if i == self.skip_connect_every:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        if self.dim_ldir > 0:
            light_xyz = input[..., -self.dim_ldir:]
            feat = torch.cat([feat, light_xyz], dim=-1)
        x = self.layers_dir[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_dir)):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        spec_coeff = self.fc_spec_coeff(x)
        diff = torch.abs(self.fc_diff(x))
        ###################################
        return normal, diff, spec_coeff
