import torch
import os
import numpy as np
from data.datamgr import SimpleDataManager
import configs
from resnet18_224_models.res18_args import parse_args
from distill.criterion import DistillKL, NCELoss, Attention, HintLoss
from resnet18_224_models.resnet18_torch import ClassificationNetwork
import torch.nn as nn
import torch.backends.cudnn as cudnn
from resnet18_224_models.train_res18 import adjust_learning_rate



def distillation_step(params, epoch, base_loader, module_list, criterion_list, optimizer):
    print_freq = 100
    avg_loss = 0
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    for i, (x, y) in enumerate(base_loader):
        # -----------Problem in training for CUB database
        # getting from 0-198 labels, converting them to 0-99
        if params.dataset == "CUB":
            y = y / 2
        x, y = x.to(params.device), y.to(params.device)
        logit_s = model_s(x)
        with torch.no_grad():
            logit_t = model_t(x)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, y)
        loss_div = criterion_div(logit_s, logit_t)

        loss_kd = 0
        loss = params.gamma * loss_cls + params.alpha * loss_div + params.beta * loss_kd

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss = avg_loss + loss.item()

        if i % print_freq == 0:
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            print(
                'Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(base_loader), avg_loss / float(i + 1)))



if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')
    image_size = 224
    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s' %(os.path.dirname(os.path.abspath(__file__)), params.dataset, 'resnet18')
    print(params.checkpoint_dir)
    print(os.path.dirname(os.path.abspath(__file__)))
    iterations = params.lr_decay_epochs.split(',')
    params.lr_decay_epochs = list([])
    for it in iterations:
        params.lr_decay_epochs.append(int(it))

    base_datamgr = SimpleDataManager(image_size, batch_size=params.batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)
    val_datamgr = SimpleDataManager(image_size, batch_size=params.test_batch_size)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.dataset == 'CUB':
        params.num_classes = 100
    elif params.dataset == 'tieredImagenet':
        params.num_classes = 351
    else:
        params.num_classes = 64

    params.file_path = '%s/checkpoints/%s/%s/best_model_rfs.tar' %(os.path.dirname(os.path.abspath(__file__)), params.dataset, 'resnet18')
    model_t = ClassificationNetwork(params)
    model_s = ClassificationNetwork(params)

    print(params.file_path)

    ckpt = torch.load(params.file_path)
    model_t.load_state_dict(ckpt['model_state_dict'])


    #data = torch.randn(2, 3, 84, 84)
    model_t.eval()
    model_s.eval()

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(params.kd_T)

    if params.distill == 'kd':
        criterion_kd = DistillKL(params.kd_T)
    elif params.distill == 'attention':
        criterion_kd = Attention()
    elif params.distill == 'hint':
        criterion_kd = HintLoss()
    else:
        raise NotImplementedError(params.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = torch.optim.SGD(trainable_list.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # routine: supervised model distillation
    for epoch in range(params.start_epoch+1, params.stop_epoch + 1):

        adjust_learning_rate(params, epoch, optimizer)
        print("==> training...")
        distillation_step(params, epoch, base_loader, module_list, criterion_list, optimizer)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)
    outfile = os.path.join(params.checkpoint_dir, 'best_model_kd.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_s.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)


