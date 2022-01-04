import numpy as np
import os
import glob
import argparse
import torch

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='miniImagenet',        help='CUB/miniImagenet/cifar/tieredImagenet')
    parser.add_argument('--model'       , default='resnet18',      help='model:  WideResNet28_10/resnet18/resnet12')
    parser.add_argument('--method'      , default='S2M2_R',   help='rotation/S2M2_R')
    parser.add_argument('--classifier' , default='LR',   help='LR/SVM//centroid')
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--lr_ft', default=1e-3, type=float, help='ft learning rate')
    parser.add_argument('--steps', default=20, type=int, help='ft steps ')
    parser.add_argument('--hidden_size', default=1024, type=int, help='hallucinator latent space size')
    parser.add_argument('--hallucinator_type', default='vector', type=str, help='vector/tensor')
    parser.add_argument('--adapt', type=bool, default=False, help='Fine-tuning the hallucinator at test time')

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=10, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=100, type=int, help ='Stopping epoch') #for meta-learning methods, each epoch contains 100 episodes. The default epoch number is dataset dependent. See train.py
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--lr'          , default=0.05, type=int, help='learning rate')
        parser.add_argument('--batch_size' , default=64, type=int, help='batch size ')
        parser.add_argument('--test_batch_size' , default=64, type=int, help='batch size ')
        parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')

        # distillation
        parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'contrast', 'hint', 'attention'])
        parser.add_argument('--trial', type=str, default='student1', help='trial id')

        parser.add_argument('-r', '--gamma', type=float, default=0.5, help='weight for classification')
        parser.add_argument('-a', '--alpha', type=float, default=0.5, help='weight balance for KD')
        parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

        # KL distillation
        parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
        # NCE distillation
        parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
        parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
        parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
        parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    elif script == 'test':
        parser.add_argument('--lr'          , default=0.05, type=int, help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='30,40', help='where to decay lr, can be a list')
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes')
        parser.add_argument('--vae_path', type=str, default='/home/michalislazarou/PhD/TFH_fewshot/gen_training', help ='')
        parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
        parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
        parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
        parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
        parser.add_argument('--n_aug_support_samples', default=0, type=int, help='The number of augmented samples for each meta test sample')
        parser.add_argument('--num_workers', type=int, default=1, metavar='N', help='Number of workers for dataloader')
        parser.add_argument('--batch_size' , default=64, type=int, help='batch size ')

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
    return args


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


