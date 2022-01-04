#------------------------------------------------------------------------------
#	Libraries
#------------------------------------------------------------------------------

#code adapted from https://github.com/thuyngch/Variational-Autoencoder-PyTorch
from collections import OrderedDict

import torch
from torch import nn
from torch import autograd

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class Flatten(nn.Module):
    def forward(self, input):
        #print(input.shape)
        #print(input.view(input.size(0), -1).shape)
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        #print(input.view(input.size(0), input.size(1), 1, 1).shape)
        return input.view(input.size(0), input.size(1), 1, 1)

class Discriminator(nn.Module):
    def __init__(self, in_dims=512, proto_dims=512):
        super().__init__()
        self.in_dims = in_dims
        self.main_module = nn.Sequential(
            nn.Linear(in_dims + proto_dims, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.b1 = 0.5
        self.b2 = 0.999
        self.critic_iter = 5
        self.lambda_term = 10

    def forward(self, z, s):
        x = torch.cat([z, s], dim=1)
        x = self.main_module(x)
        return x

#------------------------------------------------------------------------------
#  VAEGT
#------------------------------------------------------------------------------
class VAEGT(nn.Module):
    def __init__(self, in_dims=640, hid1_dims=100, num_classes=64, negative_slope=0.1, cond_latent =128, in_dec=512,  meta_cond = False, use_proto = False):
        super(VAEGT, self).__init__()
        self.in_dims = in_dims
        self.hid1_dims = hid1_dims
        self.meta_cond = meta_cond
        self.use_proto = use_proto
        if self.meta_cond:
            self.hid2_dims = cond_latent
        else :
            self.hid2_dims = in_dims
        self.num_classes = num_classes
        self.negative_slope = negative_slope
        self.in_dec = in_dec
        self.encoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(in_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, 512)),
            ('relu2', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
        ]))

        self.fc_mu = nn.Linear(512, hid1_dims)
        self.fc_var = nn.Linear(512, hid1_dims)

        # Conditioner
        self.conditioner = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(in_dims, 512)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(512, cond_latent)),
        ]))

        # Decoder
        self.decoder = nn.Sequential(OrderedDict([
            ('layer1', nn.Linear(hid1_dims+self.hid2_dims, self.in_dec)),
            ('relu1', nn.LeakyReLU(negative_slope=negative_slope, inplace=True)),
            ('layer2', nn.Linear(self.in_dec, self.in_dec)),
            ('sigmoid', nn.Sigmoid()),
        ]))

        self._init_weights()

    def forward(self, x, z):
        #if self.training:
        # Encode input
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        hx = self._reparameterize(mu, logvar)
        # --------------Encode embedding, meta-learning a conditioner----------
        if self.meta_cond:
        #conditioning on meta-learned prototypes
            hy = self.conditioner(x)
        elif self.use_proto:
        #conditioning on prototypes
            hy = z
        else:
        #conditioning on actual input feature per sample
            hy = x
        # ----------------------------------------------------------------------
        # Hidden representation
        h = torch.cat([hx, hy], dim=1)
        # Decode
        y = self.decoder(h)
        return y, mu, logvar

    def generate(self, x):
        if self.meta_cond:
            #y_onehot = self._onehot(y)
            hy = self.conditioner(x)
        else:
            hy = x
        hx = self.sample(x.shape[0]).type_as(hy)
        h = torch.cat([hx, hy], dim=1)
        y = self.decoder(h)
        return y

    def generate_fixed(self, x, z):
        if self.meta_cond:
            hy = self.conditioner(x)
        else:
            hy = x
        hx = z.type_as(z)
        h = torch.cat([hx, hy], dim=1)
        y = self.decoder(h)
        return y


    def _represent(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        hx = self._reparameterize(mu, logvar)
        return hx

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

    def _onehot(self, y):
        y_onehot = torch.FloatTensor(y.shape[0], self.num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)
        return y_onehot

    def sample(self, num_samples):
        return torch.FloatTensor(num_samples, self.hid1_dims).normal_()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class VAE_res(nn.Module):
    def __init__(self, dim=512, z_dim = 1024):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1),
            Flatten()
        )

        self.fc_mu = nn.Linear(dim//2*3*3, z_dim)
        self.fc_var = nn.Linear(dim//2*3*3, z_dim)

        self.conditioner = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim//2, kernel_size=3, stride=1),
            Flatten(),
            nn.Linear(dim//2*5*5, z_dim)
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(z_dim*2, dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x, z):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        hx = self._reparameterize(mu, logvar)
        # --------------Encode embedding, meta-learning a conditioner----------
        hy = self.conditioner(z)
        # Hidden representation
        h = torch.cat([hx, hy], dim=1)
        #print(h.shape)
        # Decode
        y = self.decoder(h)
        #print(y.shape, x.shape)
        return y, mu, logvar

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z

#------------------------------------------------------------------------------
#   Test bench
#------------------------------------------------------------------------------
if __name__ == "__main__":
    model = VAEGT()
    model.eval()

    input = torch.rand([1, 784])
    label = torch.tensor([[1]])

    output = model(input, label)
    print(output.shape)
