import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import cv2 as cv
import yaml
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.tensorboard import SummaryWriter

from models import Spherical_Gaussian, Light_Model, NeRF, NeRFSmall

from cfgnode import CfgNode
from dataloader import load_user, dataloader

from utils import writer_add_image

from draw_utils import plot_lighting
from dynamic_basis import dynamic_basis

from losses import *
import open3d as o3d

def train(input_data, training, **kwargs):
    batch_size, h, w = input_data['mask'].shape
    iters_per_epoch = None if 'iters_per_epoch' not in kwargs else kwargs['iters_per_epoch']
    iter_num = None if 'iter_num' not in kwargs else kwargs['iter_num']

    gt_rgb = input_data['rgb'].view(-1, 3).to(device)                  # (b*p,3)
    gt_shadow_mask = input_data['shadow_mask'].view(-1, 1).to(device)  # (b*p,1)
    mask = input_data['mask'][0].to(device)                            # (h, w)

    idxp = torch.where(input_data['mask'][0] > 0.5)
    num_rays = len(idxp[0])

    # (p,1), (p,3), (p,3), (p,48)
    output_depth_0, output_normal_0, output_diff_0, output_spec_coeff_0 = model()
    output_normal = output_normal_0.repeat(batch_size, 1)             # (b*p,3)
    output_diff = output_diff_0.repeat(batch_size, 1)                 # (b*p,3)
    output_spec_coeff = output_spec_coeff_0.repeat(batch_size, 1)     # (b*p,48)
    # (b*p,3), (b*p,3)
    est_light_direction, est_light_intensity = light_model.get_light_from_idx(
        idx=input_data['item_idx'].to(device))  # (n_light * n_rays, 3), (n_light * n_rays, 3)
    input_light_direction = est_light_direction

    if cfg.models.specular.type == 'Spherical_Gaussian':
        output_spec_mu = output_spec_coeff.view(-1, cfg.models.specular.num_basis, 3)  # (b*p, 12, 3)
        # dynamic_basis only valid first specular basis, and weight increase from 0->1 by steps
        if hasattr(cfg.models.specular, 'dynamic_basis'):
            if cfg.models.specular.dynamic_basis:
                output_spec_mu = dynamic_basis(output_spec_mu, epoch, end_epoch, cfg.models.specular.num_basis)

        output_spe = specular_model(light=input_light_direction, normal=output_normal,
                                    mu=output_spec_mu)  # (b*p, 12, 3)
        output_spe = output_spe.sum(dim=1)  # (b*p, 3)

    output_rho = output_diff + output_spe   # (b*p, 3), rho
    render_shading = F.relu((output_normal * input_light_direction).sum(dim=-1, keepdims=True))  # (b*p, 1), cos(n,l)
    render_rgb = output_rho * render_shading * est_light_intensity  # (b*p, 3), Li * rho * cos(n,l)

    if training:
        rgb_loss = rgb_loss_function(render_rgb * gt_shadow_mask, gt_rgb * gt_shadow_mask)
        loss = rgb_loss
        # 25% process strategy
        if epoch <= int(cfg.loss.regularize_epoches * end_epoch):  # if epoch is small, use tv to guide the network
            if cfg.loss.diff_tv_factor > 0:     # diffuse smooth
                diff_color_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
                diff_color_map[idxp] = output_diff_0
                tv_loss = totalVariation(diff_color_map, mask, num_rays) * batch_size * cfg.loss.diff_tv_factor
                loss += tv_loss
            if cfg.loss.spec_tv_factor > 0:     # specular smooth
                spec_color_map = torch.zeros((h, w, output_spec_coeff_0.size(1)), dtype=torch.float32, device=device)
                spec_color_map[idxp] = output_spec_coeff_0
                tv_loss = totalVariation(spec_color_map, mask, num_rays) * batch_size * cfg.loss.spec_tv_factor
                loss += tv_loss
            if cfg.loss.normal_tv_factor > 0:   # normal smooth
                normal_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
                normal_map[idxp] = output_normal_0
                tv_loss = totalVariation_L2(normal_map, mask, num_rays) * batch_size * cfg.loss.normal_tv_factor
                loss += tv_loss
            if cfg.loss.spec_coeff_factor > 0:  # roughness regulation
                spec_coeff_loss = F.l1_loss(output_spec_coeff_0, torch.zeros_like(output_spec_coeff_0))
                loss += spec_coeff_loss * cfg.loss.spec_coeff_factor * batch_size

        loss += regulation_light_intensity(light_model.light_intensity, writer,
                                           (epoch - 1) * iters_per_epoch + iter_num)
        loss += regulation_light_direction(light_model.light_direction_xy, writer,
                                           (epoch - 1) * iters_per_epoch + iter_num)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the running loss
        cost_t = time.time() - start_t
        est_time = cost_t / ((epoch - start_epoch) * iters_per_epoch + iter_num + 1) * (
                (end_epoch - epoch) * iters_per_epoch + iters_per_epoch - iter_num - 1)
        print(
            'epoch: %d,  iter: %2d/ %d, Training: %.4f, cost_time: %d m %2d s,  est_time: %d m %2d s' %
            (epoch, iter_num + 1, iters_per_epoch, loss.item(), cost_t // 60, cost_t % 60, est_time // 60,
             est_time % 60))
        writer.add_scalar('Training loss', rgb_loss.item(), (epoch - 1) * iters_per_epoch + iter_num)
    else:
        # save depth_map
        depth_map = np.zeros((h, w), dtype=np.float32)
        temp_dep = output_depth_0.clone().detach().cpu().numpy()  # (p, 1)
        temp_dep = -temp_dep[:, 0]
        temp_dep = 127.5 + (temp_dep - temp_dep.min()) / (temp_dep.max() - temp_dep.min()) * 127.5
        depth_map[idxp] = temp_dep
        cv.imwrite(os.path.join(log_path, 'depth.png'), depth_map.astype(np.uint8))
        writer_add_image(os.path.join(log_path, 'depth.png'), epoch, writer)

        xy = model.converter.xy.clone().detach().cpu().numpy()    # (p, 2)
        temp_dep = output_depth_0.clone().detach().cpu().numpy()  # (p, 1)
        temp_diff = output_diff_0.clone().detach().cpu().numpy()  # (p, 3)
        temp_diff = temp_diff / temp_diff.max()
        depth_map = np.zeros((h, w), dtype=np.float32)
        depth_map[idxp] = temp_dep[:, 0]
        np.save(os.path.join(log_path, "depth.npy"), depth_map)
        xyz = np.concatenate([xy*temp_dep, temp_dep], axis=-1)    # (p, 3)
        xyz = np.matmul(xyz, np.linalg.inv(K).T)                  # (p, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(temp_diff)
        o3d.io.write_point_cloud(os.path.join(log_path, "cloud.ply"), pcd)


        # save normal_map
        normal_map = torch.zeros((h, w, 3), dtype=torch.float32)
        temp_nor = output_normal_0.clone().detach().cpu()  # (p, 3)
        temp_nor[..., 1:] = -temp_nor[..., 1:]  # n_yz=-n_yz, cam_view to world view
        normal_map[idxp] = (temp_nor + 1) / 2  # (-1,1) -> (0,1)

        normal_map = normal_map.numpy()
        normal_map = (np.clip(normal_map * 255., 0, 255)).astype(np.uint8)[:, :, ::-1]  # 0-255, rgb to bgr
        cv.imwrite(os.path.join(log_path, 'normals.png'), normal_map)
        writer_add_image(os.path.join(log_path, 'normals.png'), epoch, writer)

        # save lights
        pred_ld, pred_li = light_model.get_all_lights()
        plot_lighting(os.path.join(log_path, 'lights.png'), pred_ld.cpu().numpy(), pred_li.cpu().numpy())
        np.savetxt(os.path.join(log_path, 'light_direction.txt'), pred_ld.cpu().numpy())
        np.savetxt(os.path.join(log_path, 'light_intensity.txt'), pred_li.cpu().numpy())
        writer_add_image(os.path.join(log_path, 'lights.png'), epoch, writer)

        # save diffuse
        rgb_map = torch.ones((h, w, 3), dtype=torch.float32)
        temp_diff = output_diff_0.clone().detach().cpu()
        rgb_map[idxp] = temp_diff / temp_diff.max()  # (0,1)

        rgb_map = rgb_map.numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]  # 0-255, rgb to bgr
        cv.imwrite(os.path.join(log_path, 'diffuse.png'), rgb_map)
        writer_add_image(os.path.join(log_path, 'diffuse.png'), epoch, writer)

        # save rgb
        rgb_map = torch.zeros((h, w, 3), dtype=torch.float32)
        rgb_map[idxp] = render_rgb[:len(idxp[0])].clone().detach().cpu()

        rgb_map = rgb_map.numpy()
        rgb_map = np.clip(rgb_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        cv.imwrite(os.path.join(log_path, 'rgb.png'), rgb_map)
        writer_add_image(os.path.join(log_path, 'rgb.png'), epoch, writer)

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

    # # get pre-def light init
    # if os.path.exists(os.path.join(cfg.experiment.log_path, 'light_direction.txt')) == False or \
    #         os.path.exists(os.path.join(cfg.experiment.log_path, 'light_intensity.txt')) == False:
    #     os.system('python lights_initialize.py --config {}'.format(configargs.config))

    if cfg.experiment.randomseed is not None:
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)

    device = torch.device(cfg.experiment.cuda)
    log_path = os.path.expanduser(cfg.experiment.log_path)
    data_path = os.path.expanduser(cfg.dataset.data_path)
    writer = SummaryWriter(log_path)  # tensorboard --logdir=runs
    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))
    K = np.array(cfg.camera.intrinsic).reshape(3, 3).astype(np.float32)
    K[0:2, :] = K[0:2, :] / cfg.dataset.scale
    ##########################
    # Build data loader
    input_data_dict = load_user(data_path, cfg.dataset.scale, cfg.dataset.shadow_threshold)
    training_data = dataloader(input_data_dict)
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)
    print("num_images: {}\nnum_rays: {}".format(training_data.data_len, training_data.num_valid_rays))

    iters_per_epoch = len(training_dataloader)
    ##########################


    ##########################
    # Build model
    # model = NeRF(
    #     num_layers=cfg.models.nerf.num_layers,
    #     hidden_size=cfg.models.nerf.hidden_size,
    #     skip_connect_every=cfg.models.nerf.skip_connect_every,
    #     normalized_xy=torch.from_numpy(input_data_dict['normalized_xy']),
    #     mean_var=torch.from_numpy(input_data_dict['rgb_mean_var']),
    #     xy=torch.from_numpy(input_data_dict['xy']),
    #     mask=torch.from_numpy(input_data_dict['mask']),
    #     K=torch.from_numpy(K)
    # )
    model = NeRFSmall(
        num_layers=cfg.models.nerf.num_layers,
        hidden_size=cfg.models.nerf.hidden_size,
        normalized_xy=torch.from_numpy(input_data_dict['normalized_xy']),
        mean_var=torch.from_numpy(input_data_dict['rgb_mean_var']),
        xy=torch.from_numpy(input_data_dict['xy']),
        mask=torch.from_numpy(input_data_dict['mask']),
        K=torch.from_numpy(K)
    )
    model.train()
    model.to(device)

    specular_model = Spherical_Gaussian(
        num_basis=cfg.models.specular.num_basis,
        k_low=cfg.models.specular.k_low,
        k_high=cfg.models.specular.k_high,
        trainable_k=True,
    )
    specular_model.train()
    specular_model.to(device)

    ''' light init by pretrained CNN '''
    light_init = (torch.from_numpy(np.loadtxt(os.path.join(log_path, 'light_direction.txt'), dtype=np.float32)),
                  torch.from_numpy(np.loadtxt(os.path.join(log_path, 'light_intensity.txt'), dtype=np.float32)))
    light_model = Light_Model(num_rays=np.count_nonzero(input_data_dict['mask']),
                              light_init=light_init,
                              requires_grad=True)
    light_model.train()
    light_model.to(device)

    params_list = []
    params_list.append({'params': model.parameters()})
    params_list.append({'params': specular_model.parameters()})
    params_list.append({'params': light_model.parameters()})

    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    ##########################

    if cfg.loss.rgb_loss == 'l1':
        rgb_loss_function = F.l1_loss
    elif cfg.loss.rgb_loss == 'l2':
        rgb_loss_function = F.mse_loss
    else:
        raise AttributeError('Undefined rgb loss function.')

    model.train()

    start_t = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        for iter_num, input_data in enumerate(training_dataloader):
            train(input_data=input_data, training=True, iters_per_epoch=iters_per_epoch, iter_num=iter_num)
        scheduler.step()

        if epoch % cfg.experiment.eval_every_iter == 0:
            model.eval()
            with torch.no_grad():
                train(input_data=input_data, training=False, iters_per_epoch=iters_per_epoch, iter_num=iter_num)
            model.train()

        if epoch % cfg.experiment.save_every_epoch == 0:
            savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
            torch.save({
                'global_step': epoch,
                'model_state_dict': model.state_dict(),
                'specular_model_state_dict': specular_model.state_dict(),
                'light_model_state_dict': light_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, savepath)
            print('Saved checkpoints at', savepath)
