import torch
import torch.nn.functional as F

import math

def cube_rendering(raw, pts, cnts, dx, dy, dz): # raw: output of mlp
    norm = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz)) # 立方体的对角线长度
    raw2beta = lambda raw, dists, rs, act_fn=F.relu : -act_fn(raw) * dists * torch.square(rs) * 4 * math.pi
    raw2alpha = lambda raw, dists, rs, act_fn=F.relu : (1.-torch.exp(-act_fn(raw) * dists)) * torch.square(rs) * 4 * math.pi
    # distance
    rs = torch.norm(pts - cnts[:, None], dim=-1)# 采样点到中心的距离
    sorted_rs, indices_rs = torch.sort(rs) # 排序
    dists = sorted_rs[...,1:] - sorted_rs[...,:-1] # 相邻点的距离
    dists = torch.cat([dists, dists[...,-1:]], -1)  # [N_rays, N_samples] 保持维度相同
    # color
    rgb = torch.gather(torch.sigmoid(raw[...,:-1]), -2, indices_rs[..., None].expand(raw[...,:-1].shape)) # c
    # density
    sorted_raw = torch.gather(raw[...,-1], -1, indices_rs) # σ
    beta = raw2beta(sorted_raw, dists, sorted_rs / norm)# βi=−ReLU(rawi)⋅di⋅ri^2⋅4π  di:相邻点间距, ri:到center的归一化距离
    alpha = raw2alpha(sorted_raw, dists, sorted_rs / norm)  # αi=(1−exp(−ReLU(rawi)⋅di))⋅ri^2⋅4π   #[N_rays, N_samples]
    weights = alpha * torch.exp(torch.cumsum(torch.cat([torch.zeros(alpha.shape[0], 1), beta], -1), -1)[:, :-1]) # transmittance

    rgb_map = torch.sum(weights * rgb.squeeze(), -1)

    return {'rgb' : rgb_map, 'weights' : weights, 'indices_rs' : indices_rs}