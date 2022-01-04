import sys
sys.path.append('../')
from gen_training.vaegt import VAEGT, VAE_res
from resnet18_224_models.res18_args import parse_args
from resnet18_224_models.resnet18_torch import EmbeddingNet

import os
import torch
from gen_training import vaegt_functions as vf
import torch.nn.functional as F
import configs
from data.datamgr import SetDataManager


def vae_loss(recon_x, x, mu, logvar, norm=False):
    if norm:
        recon_x = vf.scaleEachUnitaryDatas(recon_x)
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD+BCE

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def adapt(params, model, X, Y, steps=25):
    model.load_state_dict(torch.load(params.vae_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    protos, _ = vf.get_prototypes(X, Y)
    for i in range(steps):
        optimizer.zero_grad()
        loss = vf.contrastive_proto(params, model, protos, no_samples=300)
        loss.backward()
        optimizer.step()

def get_features_res18(params, model, data, feat_tensor= False, backbone='resnet18'):
    with torch.no_grad():
        n_way, _, height, width, channel = data[0].size()

        support_xs = data[0]
        support_xs = support_xs.contiguous().view(-1, height, width, channel).to(params.device)
        support_ys = data[1]
        support_ys = support_ys.contiguous().view(-1).to(params.device)
        if backbone =='resnet18':
            support_features = model(support_xs, feat_tensor=feat_tensor)
        elif backbone =='resnet12':
            support_features, _ = model(support_xs)
    return support_features, support_ys

def meta_train_vector(args, model, vae, datamgr):
    optimizer = torch.optim.Adam(vae.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    vae.train()
    for j in range(50):
        data_loader = datamgr.get_data_loader(args.loadfile, aug=True)
        for idx, data in enumerate(data_loader):
            support_features, support_ys = get_features_res18(params = args, model = model, data = data, feat_tensor=False)
            protos, _ = vf.get_prototypes(support_features, support_ys)
            optimizer.zero_grad()
            # generate from outside so that gradients propagate
            loss_cp = vf.contrastive(args, vae, protos)
            total_loss = loss_cp
            total_loss.backward()
            optimizer.step()
            if idx%200 ==0:
                print('Iteration: ', j, "Episode: ", idx,"total_loss: ", total_loss.item())
        scheduler.step()

def meta_train_tensor(args, model, vae, datamgr):
    optimizer = torch.optim.Adam(vae.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    vae.train()
    for j in range(50):
        data_loader = datamgr.get_data_loader(args.loadfile, aug=True)
        for idx, data in enumerate(data_loader):
            support_features, support_ys = get_features_res18(params = args, model = model, data = data, feat_tensor=True)
            protos, _ = vf.get_prototypes_tensor(support_features, support_ys)
            optimizer.zero_grad()
            # generate from outside so that gradients propagate
            loss_cp = vf.contrastive_proto(args, vae, protos)
            # Train
            total_loss = loss_cp
            total_loss.backward()
            optimizer.step()
            if idx%200 ==0:
                print('Iteration: ', j, "Episode: ", idx," Loss: ", total_loss.item())#, 'vae_loss: ', loss.item(), " vae_loss: ", loss_mixup.item())# 'learning rate: ', scheduler.get_lr())
        scheduler.step()

if __name__ == '__main__':
    args = parse_args('test')
    args.model = 'resnet18'
    backbone_dir = '/home/michalislazarou/PhD/TFH_fewshot/resnet18_224_models'
    if args.model == 'resnet18':
        image_size = 224
        args.n_shots = 5
        datamgr = SetDataManager(args, image_size)
        split = 'base'
        args.loadfile = configs.data_dir[args.dataset] + split + '.json'
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))
        #data_loader = datamgr.get_data_loader(args.loadfile, aug=False)
        args.file_path = '%s/checkpoints/%s/%s/best_model_kd.tar' % (backbone_dir, args.dataset, 'resnet18')
        print(args.file_path)
        print(os.path.dirname(os.path.abspath(__file__)))

        if args.dataset == 'CUB':
            args.num_classes = 100
        elif args.dataset == 'tieredImagenet':
            args.num_classes = 351
        else:
            args.num_classes = 64
        if args.hallucinator_type == 'vector':
            vaegt = VAEGT(in_dims=512, num_classes=args.num_classes, hid1_dims=args.hidden_size, cond_latent=args.hidden_size,in_dec=512, meta_cond=True).to(args.device)
            vaegt.to(args.device)
        elif args.hallucinator_type == 'tensor':
            vae_res = VAE_res()
            vae_res.to(args.device)
        model = EmbeddingNet(args).to(args.device)

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset, "vae", args.model)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if args.hallucinator_type == 'vector':
            meta_train_vector(args, model, vaegt, datamgr)
            save_file = os.path.join(save_dir, "vector.pth")
            torch.save(vaegt.state_dict(), save_file)
        elif args.hallucinator_type == 'tensor':
            meta_train_tensor(args, model, vae_res, datamgr)
            save_file = os.path.join(save_dir, "tensor.pth")
            torch.save(vae_res.state_dict(), save_file)
        print("Best checkpoint is saved at %s" % (save_file))

