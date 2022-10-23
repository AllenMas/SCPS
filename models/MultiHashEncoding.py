import torch
import torch.nn as nn

#BOX_OFFSETS = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=torch.int)
'''
0 ----- 1
|       |
|       |
2 ----- 3
'''

def hash(coords, log2_hashmap_size):
    '''
    coords: shape(n, d), this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result

def get_box_hashed_addr(xy, resolution, log2_hashmap_size):
    '''
    @param
        xy: shape(n, 2), 2D coordinates of samples, range (0, 1)
        resolution: number of pixel per axis
    @return
        box_hashed_addr: shape (n, 4), address in hash table
    '''
    top_left = torch.floor(xy * resolution).int()
    box = top_left.unsqueeze(1) + torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=torch.int, device=xy.device)
    box = torch.clip(box, 0, resolution)
    box_hashed_addr = hash(box, log2_hashmap_size)
    return box_hashed_addr


class HashEmbedder(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19,
                 base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)

        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size, self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def bilinear_interp(self, x, resolution, voxel_embedds):
        '''
        x: B x 2
        voxel_embedds: B x 4 x 2
        f0 ----- f1
        |        |
        |        |
        f2 ----- f3
        '''
        pixels = x * resolution
        weights = pixels - pixels.floor()  # B x 2
        w0, w1 = weights[:, [0]], weights[:, [1]]
        f0, f1, f2, f3 = voxel_embedds[:, 0], voxel_embedds[:, 1], voxel_embedds[:, 2], voxel_embedds[:, 3]
        f = (f0 * (1 - w0) + f1 * w0) * (1 - w1) + (f2 * (1 - w0) + f3 * w0) * w1
        return f

    def forward(self, x):
        '''
        Args:
            x: shape(n, 2), 2D coordinates of samples, range from (0, 1)
        Returns:
            x_embedded_all, shape(n, L*F)
        '''
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** i)
            box_hashed_addr = get_box_hashed_addr(x, resolution, self.log2_hashmap_size)   # (p, 4)
            voxel_embedds = self.embeddings[i](box_hashed_addr)                            # (p, 4, F)
            x_embedded = self.bilinear_interp(x, resolution, voxel_embedds)               # (p, F)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)


if __name__ == '__main__':
    size = 100
    embedder = HashEmbedder(n_levels=10, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, finest_resolution=8192)
    y, x = torch.meshgrid([torch.arange(size), torch.arange(size)])
    xy = torch.stack([x, y], dim=-1)
    xy = xy.reshape(-1, 2) / (size - 1)
    emb_xy = embedder(xy)
    print(emb_xy.shape)