import torch
import os
import numpy as np
import argparse
import yaml

from models import Light_Model_CNN
from dataloader import load_user
from cfgnode import CfgNode

def get_all_masked_images(images, mask):
    idx = torch.where(mask > 0.5)
    y_max, y_min = max(idx[0]), min(idx[0])
    x_max, x_min = max(idx[1]), min(idx[1])

    y_max, y_min = min(y_max + 15, images.shape[1]), max(y_min - 15, 0)
    x_max, x_min = min(x_max + 15, images.shape[2]), max(x_min - 15, 0)

    out_images = images[:, y_min:y_max, x_min:x_max, :].permute([0, 3, 1, 2])
    out_masks = mask[y_min:y_max, x_min:x_max][None, None, ...].repeat(out_images.size(0), 1, 1, 1)
    out = torch.cat([out_images, out_masks], dim=1)
    return out  # (num_image, 4, height, width)

def op(ckpt, device, log_path, images, mask):
    light_init_model = Light_Model_CNN(num_layers=3, hidden_size=64, output_ch=4, batchNorm=False)

    model_checkpoint_pth = os.path.expanduser(ckpt)
    ckpt = torch.load(model_checkpoint_pth)
    light_init_model.load_state_dict(ckpt['model_state_dict'])

    light_init_model.eval()
    light_init_model.to(device)

    light_init_model.set_images(
        num_rays=np.count_nonzero(mask),
        images=get_all_masked_images(torch.from_numpy(images), torch.from_numpy(mask)),
        device=device
    )
    ld, li = light_init_model.get_all_lights()
    os.makedirs(log_path, exist_ok=True)
    np.savetxt(os.path.join(log_path, 'light_direction.txt'), ld.detach().cpu().numpy())
    np.savetxt(os.path.join(log_path, 'light_intensity.txt'), li.detach().cpu().numpy())
    print("Init light info saved to {}".format(log_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diligent/reading.yml",
                        help="Path to (.yml) config file.")

    # Read config file.
    configargs = parser.parse_args()
    configargs.config = os.path.expanduser(configargs.config)
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    input_data_dict = load_user(os.path.expanduser(cfg.dataset.data_path), cfg.dataset.scale, cfg.dataset.shadow_threshold)

    device = torch.device("cuda:0")
    op(ckpt=cfg.models.light_model.load_pretrain,
       device=torch.device("cuda:0"),
       log_path=cfg.experiment.log_path,
       images=input_data_dict['images'],
       mask=input_data_dict['mask'])
