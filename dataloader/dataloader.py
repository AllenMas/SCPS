import torch
from torch.utils.data import Dataset


class dataloader(Dataset):
    def __init__(self, data_dict):
        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)  # (h, w)
        self.height = self.mask.size(0)
        self.width = self.mask.size(1)

        self.rgb = torch.tensor(data_dict['rgb'], dtype=torch.float32)  # (n, p, 3)
        self.shadow_mask = torch.tensor(data_dict['shadow_mask'], dtype=torch.float32)  # (n, p), valid/dark: 1/0

        self.data_len = self.rgb.size(0)
        self.num_valid_rays = self.rgb.size(1)

        print('DataLoader Initialized')

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        sample = {
            'mask': self.mask,
            'rgb': self.rgb[idx],                  # (p, 3)
            'shadow_mask': self.shadow_mask[idx],  # (p,), valid/dark: 1/0
            'item_idx': idx
        }
        return sample