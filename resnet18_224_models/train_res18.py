import torch
import os
import numpy as np
from data.datamgr import SimpleDataManager
import configs
from resnet18_224_models.res18_args import parse_args
from resnet18_224_models.resnet18_torch import ClassificationNetwork

def adjust_learning_rate(params, epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    #print(params.lr_decay_epochs)
    steps = np.sum(epoch > np.asarray(params.lr_decay_epochs))
    if steps > 0:
        new_lr = params.lr * (0.1 ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=5e-4)
    max_acc = 0
    model.to(params.device)

    for epoch in range(start_epoch,stop_epoch):
        adjust_learning_rate(params, epoch, optimizer)
        model.train()
        model.train_loop(epoch, base_loader,  optimizer)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

    outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, outfile)

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
    model = ClassificationNetwork(params)
    train(base_loader, val_loader, model, start_epoch, stop_epoch, params)



