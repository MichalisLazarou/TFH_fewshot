import numpy as np
import torch

class centroids:
    def __init__(self, args, alpha):
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.alpha = alpha
        self.lam = 1
        self.centroids = None

    def init_centroids(self, support, ys):
        #classes_list = torch.unique_consecutive(ys)  # .tolist()
        no_classes = torch.max(ys).detach().cpu().numpy() + 1
        Y = ys.detach().cpu().numpy()
        prototypes = []
        for i in range(no_classes):
            # for i in range(no_classes):
            idx = np.where(Y == i)
            tmp = torch.unsqueeze(torch.mean(support[idx], dim=0), 0)
            #print(tmp.shape)
            prototypes.append(tmp)
        prototypes = torch.cat(prototypes, dim=0)
        self.centroids = prototypes

    def updateFromEstimate(self, estimate):
        Dmus = estimate - self.centroids
        self.centroids = self.centroids + self.alpha * (Dmus)

    def augmentedCentroids(self, support, ys):
        classes_list = torch.unique_consecutive(ys)  # .tolist()
        # no_classes = torch.max(ys).detach().cpu().numpy() + 1
        Y = ys.detach().cpu().numpy()
        prototypes = []
        for i in classes_list:
            # for i in range(no_classes):
            idx = np.where(Y == i.item())
            tmp = torch.unsqueeze(torch.mean(support[idx], dim=0), 0)
            prototypes.append(tmp)
        prototypes = torch.cat(prototypes, dim=0)
        return prototypes


    def euclidean_distance(self, q):
        n = self.centroids.size(0)
        m = q.size(0)
        d = self.centroids.size(1)

        x = self.centroids.unsqueeze(1).expand(n, m, d)
        y = q.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(x - y, 2).sum(2)

        return dist

    def get_probs(self, dist):
        P = torch.exp(-self.lam * dist)
        u = P.sum(0)
        P /= u
        return P


    def predict(self, query):
        dist = self.euclidean_distance(query)
        P = self.get_probs(dist)
        # P = 1/dist
        query_ys_pred = np.argmax(P.detach().cpu().numpy(), 0)
        return query_ys_pred


