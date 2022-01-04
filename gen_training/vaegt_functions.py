import torch
import numpy as np
import torch.nn as nn
import os
import glob

def euclidean_distance(centroids, q):
    n = centroids.size(0)
    m = q.size(0)
    d = centroids.size(1)

    x = centroids.unsqueeze(1).expand(n, m, d)
    y = q.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    return dist

def contrastive(args, vae, protos, no_samples=10):
    z = torch.FloatTensor(no_samples, vae.hid1_dims).normal_().to(args.device)
    protos_repeat = torch.repeat_interleave(protos, repeats=no_samples, dim=0)
    z = torch.cat(len(protos)*[z], dim=0)
    mse = nn.MSELoss()
    #minimize MSE between same protos
    if vae.meta_cond:
        hy = vae.conditioner(protos_repeat)
    else:
        hy = protos_repeat
    h = torch.cat([z, hy], dim=1)
    anchor = vae.decoder(h)
    loss = mse(anchor, protos_repeat)
    return loss

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def contrastive_proto(args, vae, protos, no_samples=10):
    z = torch.FloatTensor(no_samples, vae.z_dim).normal_().to(args.device)
    protos_repeat = torch.repeat_interleave(protos, repeats=no_samples, dim=0)
    z = torch.cat(len(protos)*[z], dim=0)
    mse = nn.MSELoss()
    #minimize MSE between same protos
    hy = vae.conditioner(protos_repeat)
    h = torch.cat([z, hy], dim=1)
    anchor = vae.decoder(h)
    loss = mse(anchor, protos_repeat)
    return loss

def adapt_tensor(params, vae, support_features, support_ys, no_samples=10):
    optimizer = torch.optim.Adam(vae.parameters(), lr = params.lr_ft)
    protos, _ = get_prototypes_tensor(support_features, support_ys)
    for i in range(params.steps):
        loss_cp = contrastive_proto(params, vae, protos, no_samples=no_samples)
        optimizer.zero_grad()
        loss_cp.backward()
        optimizer.step()

def augment_tensor(params, vaegt, support_features, support_ys, no_samples=500):
    z = torch.FloatTensor(no_samples, vaegt.z_dim).normal_().to(params.device)
    ys = torch.unique(support_ys)
    protos, _ = get_prototypes_tensor(support_features, support_ys)
    protos_repeat, aug_ys = torch.repeat_interleave(protos, repeats=no_samples, dim=0), torch.repeat_interleave(ys, repeats=no_samples, dim=0)
    if params.adapt == True:
        adapt_tensor(params, vaegt, support_features, support_ys, 10)
    z = torch.cat(len(protos)*[z], dim=0)
    hy = vaegt.conditioner(protos_repeat)
    h = torch.cat([z, hy], dim=1)
    aug_fs = vaegt.decoder(h)
    return aug_fs, aug_ys

def augment_vector(params, vaegt, support_features, support_ys, no_samples=500):
    z = torch.FloatTensor(no_samples, vaegt.hid1_dims).normal_().to(params.device)
    ys = torch.unique(support_ys)
    protos, _ = get_prototypes_tensor(support_features, support_ys)
    protos_repeat, aug_ys = torch.repeat_interleave(protos, repeats=no_samples, dim=0), torch.repeat_interleave(ys, repeats=no_samples, dim=0)
    z = torch.cat(len(protos)*[z], dim=0)
    if vaegt.meta_cond:
        hy = vaegt.conditioner(protos_repeat)
    else:
        hy = protos_repeat
    h = torch.cat([z, hy], dim=1)
    aug_fs = vaegt.decoder(h)
    return aug_fs, aug_ys

def get_prototypes(fs, ys):
    classes_list = torch.unique(ys)#.tolist()
    #no_classes = torch.max(ys).detach().cpu().numpy() + 1
    Y = ys.detach().cpu().numpy()
    prototypes = []
    for i in classes_list:
    #for i in range(no_classes):
        idx = np.where(Y == i.item())
        tmp = torch.unsqueeze(torch.mean(fs[idx], dim=0), 0)
        prototypes.append(tmp)
    prototypes = torch.cat(prototypes, dim = 0)
    return prototypes, classes_list

def get_prototypes_tensor(fs, ys):
    classes_list = torch.unique(ys)#.tolist()
    Y = ys.detach().cpu().numpy()
    prototypes = []
    for i in classes_list:
        idx = np.where(Y == i.item())
        tmp = torch.mean(fs[idx], dim=0, keepdim=True)
        prototypes.append(tmp)
    prototypes = torch.cat(prototypes, dim = 0)
    return prototypes, classes_list


def scaleEachUnitaryDatas(datas):
    norms = datas.norm(dim=1, keepdim=True)
    return datas / norms
