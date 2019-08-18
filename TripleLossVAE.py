import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
# 可以参考
# https://github.com/hmishfaq/DDSM-TVAE
CUDA = True
LR = 1e-4
TRIPLET_LOSS = 3
EMBED_LOSS = 5e-3
MARGIN = 0.2  # 'margin for triplet loss (default: 0.2)')
BETA1 = 0.9
BETA2 = 0.999
VAE_LOSS = 1
latent_dim = 8
log_interval = 30
train_loss_metric = []
train_loss_VAE = []
train_acc_metric = []


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r

class Encoder(torch.nn.Module):  #
    def __init__(self, D_in, H, D_out):  # input dim | 100 | 100
        super(Encoder, self).__init__()
        # 网络层
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        # 隐层信息
        self.mean =torch.nn.Linear(D_out, latent_dim)
        self.logvar =torch.nn.Linear(D_out, latent_dim)

    def sampler(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        if CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)  # 和之前VAE的效果一样

    def forward(self, x):
        batch_size = x.size(0) # 一下子传多个样本
        x = F.relu(self.linear1(x))  # 非常简单的线性激活网络，以后要修改
        x = F.relu(self.linear2(x))  # 以后需要对这个进行改变 适应不同信息
        # 在TVAE层中 这层进行采样
        hidden = x
        hidden = hidden.view(batch_size, -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        #return F.relu(self.linear2(x))
        return latent_z, mean, logvar

class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        batch_size = x.size(0)  # batch_size 以后是否会用到
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

# 一般的VAE 网络
class VAE(torch.nn.Module):
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)  # 这个是 mu 层
        self._enc_log_sigma = torch.nn.Linear(100, 8)  # 这个是 logvar 层

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda(0)

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)

        z = self._sample_latent(h_enc)
        return self.decoder(z)

# Triple Net 网络
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        #print (x.shape)
        #print(y.shape)
        #print(z.shape)
        latent_x,mean_x,logvar_x = self.embeddingnet(x)
        latent_y,mean_y,logvar_y = self.embeddingnet(y)
        latent_z,mean_z,logvar_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(mean_x, mean_y, 2)  # a 和 p 距离
        dist_b = F.pairwise_distance(mean_x, mean_z, 2)  # a 和 n 距离
        return latent_x,mean_x,logvar_x,\
            latent_y,mean_y,logvar_y,\
            latent_z,mean_z,logvar_z,\
            dist_a, dist_b
# loss 设定
#loss functions
mse = nn.MSELoss()
kld_criterion = nn.KLDivLoss()

#reconstrunction loss  重构loss
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())
    return fpl

def loss_function(recon_x,x,mu,logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    #target_feature = descriptor(x)  # 这玩意就是网络 # 也可以直接使用encoder 和 decoder
    #recon_features = descriptor(recon_x)
    #FPL = fpl_criterion(recon_features, target_feature)
    FPL = 0.0 # 先暂时不用这个
    return KLD+0.5*FPL

def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

# 正确率 和 学习率 的辅助函数
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(dist_a, dist_b):
    margin = 0
    #pred = (dist_a - dist_b - margin).cpu().data  # 这个是写错了把 HERE
    pred = (dist_b - dist_a).cpu().data  # 这个是写错了把 HERE

    tmp = float(((pred > 0).sum()*1.0).cpu().item())  # pred 大于0 即正确
    tmp2 = float(dist_a.size()[0])  # 全部的数据
    return (tmp/tmp2)

def accuracy_id(dist_a, dist_b, c, c_id):
    margin = 0
    pred = (dist_a - dist_b - margin).cpu().data
    return ((pred > 0)*(c.cpu().data == c_id)).sum()*1.0/(c.cpu().data == c_id).sum()


# 进行train构建
def train(train_loader, tnet, decoder, criterion, optimizer, epoch):
    '''

    :param train_loader:
    :param tnet:  triple loss 的encoder
    :param decoder:
    :param criterion:
    :param optimizer:
    :param epoch:
    :return:
    '''
    losses_metric = AverageMeter()
    losses_VAE = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()


    tnet.train()  # 对TVAE 的 encoder 训练
    decoder.train()  # 对TVAE 的 decoder 训练
    #for batch_idx, (data1, data2, data3) in enumerate(train_loader):
    for batch_idx, (datas) in enumerate(train_loader):
        data1 = datas[0]
        data2 = datas[1]
        data3 = datas[2]
        if CUDA:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        if data1.shape[0] == 0: # 这里硬循环
            break
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
        latent_x,mean_x,logvar_x,latent_y,mean_y,logvar_y,latent_z,mean_z,logvar_z,dist_a, dist_b = tnet(data1, data2, data3)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if CUDA:
            target = target.cuda()
        target = Variable(target)
        #get reconstructed images
        reconstructed_x = decoder(latent_x)
        reconstructed_y = decoder(latent_y)
        reconstructed_z = decoder(latent_z)
        # 对 loss function 的 descriptor 是否需要？删掉
        loss_vae = loss_function(reconstructed_x, data1, mean_x, logvar_x)
        loss_vae += loss_function(reconstructed_z, data2, mean_y, logvar_y)
        loss_vae += loss_function(reconstructed_z, data3, mean_z, logvar_z)
        loss_vae = loss_vae/(3*len(data1))
        # triplet 的方法
        # target - vec of 1. This is what i want : dista >distb = True
        #loss_triplet = criterion(dist_a, dist_b, target)  # 感觉 这个loss用错了
        loss_triplet = criterion(mean_x, mean_y, mean_z)  # 感觉 这个loss用错了

        # dist_a 是 a 和 p 的距离， dist_b是a 和 n 的距离
        loss_embedd = mean_x.norm(2) + mean_y.norm(2) + mean_z.norm(2)
        loss =TRIPLET_LOSS*loss_triplet + EMBED_LOSS * loss_embedd + VAE_LOSS*loss_vae
        acc = accuracy(dist_a, dist_b)
        #         losses_metric.update(loss_triplet.data[0], data1.size(0))
        losses_metric.update(loss_triplet.item(), data1.size(0))
        #         losses_VAE.update(loss_vae.data[0], data1.size(0))
        losses_VAE.update(loss_vae.data.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data.item()/3, data1.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if batch_idx % log_interval == 0:
        if batch_idx % log_interval == 0:
            print('batch_idx: {}\t'
                  'Train Epoch: {} [{}/{}]\t'
                  'VAE Loss: {:.4f} ({:.4f}) \t'
                  'Metric Loss: {:.4f} ({:.4f}) \t'
                  'Metric Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                batch_idx,
                epoch, batch_idx * len(data1), (train_loader.N),
                losses_VAE.val, losses_VAE.avg,  # 全局损失
                losses_metric.val, losses_metric.avg,  # 度量损失
                100. * accs.val, 100. * accs.avg,  # 度量的正确率
                emb_norms.val, emb_norms.avg))  # 是否靠近均值0的空间

            train_loss_metric.append(losses_metric.val)
            train_loss_VAE.append(losses_VAE.val)
            train_acc_metric.append(accs.val)


if __name__ == '__main__':
    '''
    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST('D:/CV/data', download=True, transform=transform)

    #dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
    #                                         shuffle=True, num_workers=2)
    y = (train_data.targets > 4)
    #y = (train_data.targets > 4).astype(np.int) # 进行两类拆分
    #exit()
    train_data.targets = y
    # 需要对 data Loader 进行修改
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    
    '''
    input_dim = 28 * 28
    import MINST_triple_dataloader as Mt
    train_loader = Mt.TripletImageLoader(dataShape = input_dim)
    # 初始化 encoder
    encoder = Encoder(input_dim, 100, 8)
    print (encoder)
    if CUDA:
        encoder = encoder.cuda()
    tnet = Tripletnet(encoder)
    print (tnet)
    if CUDA:
        tnet.cuda()
    decoder = Decoder(8, 100, input_dim)
    print (decoder)
    if CUDA:
        decoder.cuda()
    #criterion = torch.nn.MarginRankingLoss(margin = MARGIN)  # 这个是 度量学习 loss？
    criterion = torch.nn.TripletMarginLoss(margin = MARGIN, reduce='sum')

    parameters = list(tnet.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=LR, betas=(BETA1, BETA2))
    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])

    print('  + Number of params in tnet: {}'.format(n_parameters))
    # 暂时只进行 Train
    for epoch in range(0, 200):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, tnet,decoder, criterion, optimizer, epoch)

    exit()
    '''
    print('Number of samples: ', len(train_data))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(8, 100, input_dim)
    vae = VAE(encoder, decoder).cuda(0)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            inputs = inputs.cuda(0)
            classes = classes.cuda(0)
            optimizer.zero_grad()
            dec = vae(inputs)
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll # 这
            loss.backward()
            optimizer.step()
            l = loss.data.item()
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].cpu().numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
    '''

