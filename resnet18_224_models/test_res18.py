import sys
sys.path.append('../')
import numpy as np
import configs
import random
import os
#import tkinter
import matplotlib

from gen_training.vaegt import VAEGT, VAE_res
from resnet18_224_models.res18_args import parse_args
from resnet18_224_models.resnet18_torch import EmbeddingNet
from data.datamgr import SetDataManager, DTNManager
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import t
#from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from centroids_class import centroids

#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch
from gen_training import vaegt_functions as vf



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def convert_to_few_shot_labels(data):
    for i in range(data.shape[0]):
        data[i,:] = i
    return data

def get_predictions(params, model, data, vae = 'nothing', vaegt = None, prior=None):
    with torch.no_grad():
        data[1] = convert_to_few_shot_labels(data[1])
        n_way, _, height, width, channel = data[0].size()

        support_xs, query_xs = data[0][:, :params.n_shots], data[0][:, params.n_shots:]
        support_ys, query_ys = data[1][:, :params.n_shots], data[1][:, params.n_shots:]
        support_xs = support_xs.contiguous().view(-1, height, width, channel).to(params.device)
        query_xs = query_xs.contiguous().view(-1, height, width, channel).to(params.device)
        support_ys = support_ys.contiguous().view(-1)
        query_ys = query_ys.contiguous().view(-1)
        if vae == 'vae_tensor':
            support_features, query_features = model(support_xs, feat_tensor = True), model(query_xs, feat_tensor = True)
        else:
            support_features, query_features = model(support_xs), model(query_xs)

    if vae == 'vae_vector':
        vaegt.eval()
        vaegt.load_state_dict(torch.load(params.vae_path))
        aug_features, aug_ys = vf.augment_vae_same(params, vaegt, support_features, support_ys, no_samples=100)
        aug_features, aug_ys, support_features, support_ys = aug_features.cpu(), aug_ys.cpu(), support_features.cpu(), support_ys.cpu()
        support_features, support_ys = torch.cat((support_features, aug_features), dim=0).cpu(), torch.cat( (support_ys, aug_ys), dim=0).cpu()
        query_features, query_ys = query_features.cpu(), query_ys.cpu()
    elif vae == 'vae_tensor':
        vaegt.eval()
        vaegt.load_state_dict(torch.load(params.vae_path))
        aug_features, aug_ys = vf.augment_vae_tensor(params, vaegt, support_features, support_ys, no_samples=100)
        aug_features, support_features= model.avgpool(aug_features), model.avgpool(support_features)
        query_features = model.avgpool(query_features)
        aug_features, support_features, query_features = aug_features.view(aug_features.size(0), -1), support_features.view(support_features.size(0), -1), query_features.view(query_features.size(0), -1)
        aug_features, aug_ys, support_features, support_ys = aug_features.cpu(), aug_ys.cpu(), support_features.cpu(), support_ys.cpu()
        support_features, support_ys = torch.cat((support_features, aug_features), dim=0), torch.cat( (support_ys, aug_ys), dim=0)

    support_features, query_features = support_features.detach().cpu(), query_features.detach().cpu()
    support_ys, query_ys = support_ys.detach().cpu(), query_ys.detach().cpu()

    if params.classifier == 'LR':
        clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    elif params.classifier == 'SVM':
        clf = SVC(C=10, gamma='auto', kernel='linear', probability=True)
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    elif params.classifier == 'centroid':
        clf = centroids(params, alpha=0.5)
        clf.init_centroids(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)
    return query_ys, query_ys_pred



def few_shot_test(params, model, testloader, vae='nothing'):
    model = model.eval()
    model.to(params.device)
    acc = []
    if vae == 'vae_vector':
        # using vae with commenting the original code
        vaegt = VAEGT(in_dims=512, num_classes=params.num_classes, hid1_dims=params.hidden_size,cond_latent=params.hidden_size, in_dec=512, meta_cond=True).to(params.device)
        vaegt.load_state_dict(torch.load(params.vae_path))
    elif vae == 'vae_tensor':
        vaegt = VAE_res().to(params.device)
        vaegt.load_state_dict(torch.load(params.vae_path))
    else:
        vaegt = None
    for idx, data in tqdm(enumerate(testloader)):
        query_ys, query_ys_pred = get_predictions(params, model, data, vae, vaegt)
        acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
    return mean_confidence_interval(acc)

if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    print(plt.get_backend())
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    params = parse_args('test')
    image_size = 224
    datamgr = SetDataManager(params, image_size)
    split = 'novel'

    loadfile = configs.data_dir[params.dataset] + split + '.json'
    data_loader = datamgr.get_data_loader(loadfile, aug=False)
    print(split)
    # only for cross-few-shot uncomment
    #params.dataset = 'miniImagenet'
    params.hallucinator_dir = '/home/michalislazarou/PhD/TFH_fewshot/gen_training'
    params.file_path = '%s/checkpoints/%s/%s/best_model_kd.tar' %(os.path.dirname(os.path.abspath(__file__)), params.dataset, 'resnet18')
    print(params.file_path)
    print(os.path.dirname(os.path.abspath(__file__)))

    if params.dataset == 'CUB':
        params.num_classes = 100
    elif params.dataset == 'tieredImagenet':
        params.num_classes = 351
    else:
        params.num_classes = 64
    model = EmbeddingNet(params)
    method = 'inductive_tensor' # 'inductive_tensor' is for the WACV paper
    # method = 'inductive_LR'
    if method == 'inductive':
        test_acc, test_std = few_shot_test(params, model, data_loader, vae='nothing')
    elif method == 'inductive_vector':
        params.vae_path = '%s/%s/%s/%s/%s' % (params.hallucinator_dir, params.dataset, 'vae', params.model, "vector.pth")
        print(params.vae_path)
        test_acc, test_std = few_shot_test(params, model, data_loader, vae='vae_vector')
    elif method == 'inductive_tensor':
        params.vae_path = '%s/%s/%s/%s/%s' % (params.hallucinator_dir, params.dataset, 'vae', params.model, 'tensor.pth')
        print(params.vae_path)
        test_acc, test_std = few_shot_test(params, model, data_loader, vae='vae_tensor')


    print("Classifier: {}, shots: {}, dataset: {}, acc: {:0.2f} +- {:0.2f}, lr_ft:{}, steps: {}".format(params.classifier, params.n_shots, params.dataset, test_acc*100, test_std*100, params.lr_ft, params.steps))