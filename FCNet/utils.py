import h5py
import torch
import shutil
import numpy as np
import os
import torch.nn.functional as F

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, output_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(output_path,filename))
    if is_best:
        print("saving_best")
        torch.save(state, os.path.join(output_path,'model_best.pth.tar'))
        #shutil.copyfile(filename, os.path.join(output_path,'model_best.pth.tar'))

def gt_orth_loss(uu_feature, zz_feature):
    zz_cfeature = zz_feature.view(zz_feature.shape[0], zz_feature.shape[1], -1)
    uu_cfeature = uu_feature.view(uu_feature.shape[0], uu_feature.shape[1], -1)
    orth_pre = torch.bmm(zz_cfeature.transpose(1, 2), uu_cfeature)
    orth_loss = 0.0001 * torch.sum(torch.pow(torch.diagonal(orth_pre, dim1=-2, dim2=-1), 2))

    return orth_loss

def gt_sim_loss(u_feature, z_feature, uu_feature, zz_feature):
#     z_cfeature = zz_feature.view(z_feature.shape[0], z_feature.shape[1], -1)
#     zz_cfeature = zz_feature.view(zz_feature.shape[0], zz_feature.shape[1], -1)

#     recon_sim = torch.bmm(zz_cfeature.transpose(1, 2), z_cfeature)
#     sim_gt = torch.linspace(0, z_feature.shape[2] * z_feature.shape[3] - 1,
#                             z_feature.shape[2] * z_feature.shape[3]).unsqueeze(0).repeat(z_feature.shape[0],
#                                                                                          1).cuda()  # [2,6400]
#     sim_loss = F.cross_entropy(recon_sim, sim_gt.long(), reduction='none') * 0.1

    u_cfeature = uu_feature.view(u_feature.shape[0], u_feature.shape[1], -1)
    uu_cfeature = uu_feature.view(uu_feature.shape[0], uu_feature.shape[1], -1)
    recon_sim_2 = torch.bmm(uu_cfeature.transpose(1, 2), u_cfeature)
    sim_gt_2 = torch.linspace(0, u_feature.shape[2] * u_feature.shape[3] - 1,
                              u_feature.shape[2] * u_feature.shape[3]).unsqueeze(0).repeat(u_feature.shape[0],
                                                                                           1).cuda()  # [2,6400]
    sim_loss_2 = F.cross_entropy(recon_sim_2, sim_gt_2.long(), reduction='none') * 0.1

    return sim_loss_2

def get_sim_loss(feature1, feature2):
    cos_sim = F.cosine_similarity(feature1, feature2, dim=1)
    
    # 定义损失函数，使相似性逐渐相近
    loss = (1 - cos_sim.mean())
    
    
    return loss

def get_sep_loss(feature1, feature2):
    cos_sim = F.cosine_similarity(feature1, feature2, dim=1)
    
    # 定义损失函数，使相似性逐渐降低
    loss = cos_sim.mean()
    
    return loss