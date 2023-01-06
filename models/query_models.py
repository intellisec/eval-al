import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        assert len(feature_sizes) == len(num_channels)
        
        self.gap_layers = nn.ModuleList()
        self.fc_layers  = nn.ModuleList()
        for f, n in zip(feature_sizes, num_channels) :
            self.gap_layers.append(nn.AvgPool2d(f))
            self.fc_layers.append(nn.Linear(n, interm_dim))

        self.linear = nn.Linear(len(feature_sizes) * interm_dim, 1)
    
    def forward(self, features):
        xs = []
        for x, gap, fc in zip(features, self.gap_layers, self.fc_layers) :
            x = gap(x)
            x = x.view(x.size(0), -1)
            x = F.relu(fc(x))
            xs.append(x)

        out = self.linear(torch.cat(xs, 1))
        return out

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)        
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)
        
class InnerProductDecoder(nn.Module):
    """Decoder for use Inner product for prediction"""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z,z.t()))
        return adj

class GC_VAE(nn.Module):
    """Graph convolutional based encoder and decoder"""
    def __init__(self, gc_nfeat, gc_nhid, gc_dropout, z_dim=32, nc=3, f_filt=4):
        super(GC_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.dropout = gc_dropout
        self.gc0 = GraphConvolution(gc_nfeat,gc_nhid)
        self.gc1 = GraphConvolution(gc_nhid,z_dim)
        self.gc2 = GraphConvolution(gc_nhid,z_dim)
        self.dc = InnerProductDecoder(gc_dropout)
    def encoder(self,x, adj):
        x = F.relu(self.gc0(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        mu = F.relu(self.gc1(feat, adj))  
        logvar = F.relu(self.gc2(feat, adj))
        return mu, logvar
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def forward(self, x, adj):
        mu,logvar = self.encoder(x, adj)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.dc(z)
        return adj_recon, z, mu, logvar
    


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
        
class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4, x_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.x_dim = x_dim
        fc_size = 1024 * (x_dim // 2**4)**2

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, fc_size)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(fc_size, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(fc_size, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, fc_size * 2 * 2),                           # B, 1024*8*8
            View((-1, 1024, x_dim // 2**3, x_dim // 2**3)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, device):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar, device)
        x_recon = self._decode(z)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar, device):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            with torch.cuda.device(device):
                stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, ranker_code=0):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim+ranker_code, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self,z):
    
        # zr= torch.cat([z, y], dim =1)
        # print('size after cat ranker code {}, before {}'.format(zr.shape, z.shape))
        return self.net(z)

class CDiscriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10, ranker_code=1):
        super(CDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim+ranker_code, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, y, z):
    
        zr= torch.cat([z, y], dim =1)
        # print('size after cat ranker code {}, before {}'.format(zr.shape, z.shape))
        return self.net(zr)
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class CVAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, f_filt=4, ranker_code=1, x_dim=32):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.f_filt = f_filt
        self.x_dim = x_dim
        fc_size = 1024 * (x_dim // 2**4)**2

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, self.f_filt, 2, 1, bias=False),            # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, fc_size)),                                 # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(fc_size, z_dim)                            # B, z_dim
        self.fc_logvar = nn.Linear(fc_size, z_dim)                            # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim+ranker_code, fc_size * 2 * 2),                           # B, 1024*8*8
            View((-1, 1024, x_dim // 2**3, x_dim // 2**3)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, self.f_filt, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self,y,x, device):

        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar, device)
        x_recon = self._decode(z, y)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar,device):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            with torch.cuda.device(device):
                stds, epsilon = stds.cuda(), epsilon.cuda()
    
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z, y):
        zr= torch.cat([z, y], dim =1)
        # print('size after cat ranker code {}, before {}'.format(zr.shape, z.shape))
        return self.decoder(zr)

class FCVAE(VAE):
    """Non-convolutional VAE variant"""
    def __init__(self, z_dim=32, x_dim=30000) :
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        fc_size = 512

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, fc_size * 8),
            nn.ReLU(True),
            nn.Linear(fc_size * 8, fc_size * 4),
            nn.ReLU(True),
            nn.Linear(fc_size * 4, fc_size * 2),
            nn.ReLU(True),
            nn.Linear(fc_size * 2, fc_size),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(fc_size, z_dim)
        self.fc_logvar = nn.Linear(fc_size, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, fc_size * 2),
            nn.ReLU(True),
            nn.Linear(fc_size * 2, fc_size * 4),
            nn.ReLU(True),
            nn.Linear(fc_size * 4, fc_size * 8),
            nn.ReLU(True),
            nn.Linear(fc_size * 8, x_dim)
        )
        self.weight_init()

class FCCVAE(CVAE):
    """Non-convolutional VAE variant"""
    def __init__(self, z_dim=32, x_dim=30000, ranker_code=1) :
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        fc_size = 512

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, fc_size * 8),
            nn.ReLU(True),
            nn.Linear(fc_size * 8, fc_size * 4),
            nn.ReLU(True),
            nn.Linear(fc_size * 4, fc_size * 2),
            nn.ReLU(True),
            nn.Linear(fc_size * 2, fc_size),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(fc_size, z_dim)
        self.fc_logvar = nn.Linear(fc_size, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + ranker_code, fc_size),
            nn.ReLU(True),
            nn.Linear(fc_size, fc_size * 2),
            nn.ReLU(True),
            nn.Linear(fc_size * 2, fc_size * 4),
            nn.ReLU(True),
            nn.Linear(fc_size * 4, fc_size * 8),
            nn.ReLU(True),
            nn.Linear(fc_size * 8, x_dim)
        )
        self.weight_init()