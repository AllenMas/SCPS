import torch

def regulation_circle(light_direction, writer, steps, gamma=100):
    # circle regulation
    mu = torch.mean(light_direction, dim=0)
    r = torch.sum(torch.square(light_direction - mu), dim=-1)   # (144,)
    r = torch.stack([r[0::4], r[1::4], r[2::4], r[3::4]])       # (4, 36), 4 circle with 36 lights
    r_ = torch.mean(r, dim=-1, keepdim=True)                    # (4, 1), 4 circles' radius
    loss_ = torch.sum(torch.abs(r - r_))
    writer.add_scalar('reg_direction_circle', loss_, steps)
    loss__ = torch.sum(torch.exp(-gamma*torch.diff(r_.squeeze())))
    writer.add_scalar('reg_direction_multi_circle', loss__, steps)
    return loss_ + loss__

def regulation_light_direction(light_direction, writer, steps):
    '''
        @param
        light_direction: shape (num_lights, 2)
    '''
    # mu_x = mu_y = 0
    mu = torch.mean(light_direction, dim=0)
    loss = torch.sum(torch.abs(mu))
    writer.add_scalar('reg_direction_mu', loss, steps)
    # var_x = var_y
    var = torch.std(light_direction, dim=0)
    writer.add_scalar('reg_direction_std', torch.abs(var[0] - var[1]), steps)
    loss += torch.abs(var[0] - var[1])
    # cov(x, y) = 0
    cov = torch.mean(light_direction[:, 0] * light_direction[:, 1]) - mu[0] * mu[1]
    writer.add_scalar('reg_direction_cov', torch.abs(cov), steps)
    loss += torch.abs(cov)
    return loss

def regulation_light_intensity(light_intensity, writer, steps):
    '''
        @param
        light_intensity: shape (num_lights, 1)
    '''
    loss = torch.std(light_intensity, dim=0)[0]

    writer.add_scalar('reg_intensity', loss, steps)
    return loss

def Fresnel_Factor(light, half, view, normal):
    c = torch.abs((light * half).sum(dim=-1))
    g = torch.sqrt(1.33**2 + c**2 - 1)
    temp = (c*(g+c)-1)**2 / (c*(g-c)+1)**2
    f = (g-c)**2 / (2*(g+c)**2) * (1 + temp)
    return f


def totalVariation(image, mask, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]   # grad_y
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]   # grad_x
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var


def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]  # grad_y^2
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]  # grad_x^2
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var