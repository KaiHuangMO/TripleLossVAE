from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import scipy.io as sio
import torch
from scipy import misc
import torchvision as vision
import torch.utils.data as Data

# image loader for training and testing with mean and std for pixel values
# 这个进行了数据归一化，需要思考Finance数据是否需要
def default_image_loader(path):
    img = misc.imread(path)
    imgn = np.where(img>0,img,np.nan)
    mean = np.nanmean(imgn,axis=(0,1))
    std =np.nanstd(imgn,axis=(0,1))
    return img, mean, std

class TripletImageLoader(): # 先尝试 MNIST
    def __init__(self, dataShape, batchSize=16):
        transform = transforms.Compose(
            [transforms.ToTensor()])
        #self.train_data = torchvision.datasets.MNIST('D:/CV/data', download=True, transform=transform)
        self.train_data = torchvision.datasets.MNIST('D:/CV/data', download=True)
        #self.train_data = torchvision.datasets.MNIST('./', download=True)

        #train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        y = (self.train_data.targets > 8) # 可以在这里进行样本倾斜 8作为MNIST的测试
        self.train_data.targets = y
        # 类别的index
        self.minClass = [i for i, j in enumerate(self.train_data.targets) if j == 1]  # 暂定1 为少数类
        self.majorClass = [i for i, j in enumerate(self.train_data.targets) if j == 0]
        self.N = len(self.minClass)
        self.batchSize = batchSize
        self.dataShape = dataShape

    # gets the number of batches this generator returns
    # 这个直接改变接下去的 getitem
    def __len__(self):
        l, rem = divmod(self.N, self.batchSize)
        return (l + (1 if rem > 0 else 0))

    '''
    def __getitem__(self, i):    # 可以以少类为轴，寻找多类
        z = 1
        start = i * self.batchSize
        stop = np.min([(i + 1) * self.batchSize, self.N])  # clip stop index to be <= N
        # Memory preallocation
        #ANCHOR = np.zeros((stop - start,) + self.shape + (3,))
        #POSITIVE = np.zeros((stop - start,) + self.shape + (3,))
        #NEGATIVE = np.zeros((stop - start,) + self.shape + (3,))
        # pos数据
        pos_idx = (self.minClass)  # 直接寻找 y

        # pos_idx_hat = pos_idx[(pos_idx < start) | (pos_idx > stop)]
        pos_idx_hat = []
        for ls in range(0, len(pos_idx)):
            if ls < start or ls > stop:
                pos_idx_hat.append(ls)

        ancor_labels = self.minClass[start:stop] # 标签
        # 数据
        ancor_datas = []
        for j in ancor_labels:
            ancor_datas.append(self.train_data.data[j].view(self.dataShape).tolist())

        pos_datas = []
        neg_datas = []
        for k, label in enumerate(ancor_labels):
            neg_idx = np.where(self.majorClass)  # 负样本
            neg_idx = (self.majorClass)  # 负样本
            neg_datas.append( self.train_data.data[np.random.choice(neg_idx)].view(self.dataShape).tolist())
            # pos 样本
            #pos_idx = np.where(self.minClass)[0] # 直接寻找 y


            if len(pos_idx_hat):
                pos_datas.append( self.train_data.data[np.random.choice(pos_idx_hat)].view(self.dataShape).tolist())
            else:
                # positive examples are within the batch or just 1 example in dataset
                pos_datas.append( self.train_data.data[np.random.choice(pos_idx)].view(self.dataShape).tolist())
        # 加入多数类别 尝试
        neg_idx = (self.minClass)  # 负样本 为少数类
        anr_idx = (self.majorClass)  # 正样本 为多数类
        for i in range(0, len(ancor_labels)):

            neg_datas.append(self.train_data.data[np.random.choice(neg_idx)].view(self.dataShape).tolist())


            ancor_datas.append(self.train_data.data[np.random.choice(anr_idx)].view(self.dataShape).tolist())
            pos_datas.append(self.train_data.data[np.random.choice(anr_idx)].view(self.dataShape).tolist())

        #

        ancor_datas = torch.FloatTensor(ancor_datas)
        neg_datas = torch.FloatTensor(neg_datas)
        pos_datas = torch.FloatTensor(pos_datas)

        # 对于MNIST 直接赋值
        ANCHOR = (ancor_datas)
        POSITIVE = (pos_datas)
        NEGATIVE = (neg_datas)
        return [ANCHOR, POSITIVE, NEGATIVE]

    '''
    # 似乎这样做会比较收敛 在准确率上会正确
    def __getitem__(self, i):    # 可以以少类为轴，寻找多类
        z = 1
        start = i * self.batchSize
        stop = np.min([(i + 1) * self.batchSize, self.N])  # clip stop index to be <= N
        # Memory preallocation
        #ANCHOR = np.zeros((stop - start,) + self.shape + (3,))
        #POSITIVE = np.zeros((stop - start,) + self.shape + (3,))
        #NEGATIVE = np.zeros((stop - start,) + self.shape + (3,))
        # pos数据
        pos_idx = (self.minClass)  # 直接寻找 y

        # pos_idx_hat = pos_idx[(pos_idx < start) | (pos_idx > stop)]
        pos_idx_hat = []
        for ls in range(0, len(pos_idx)):
            if ls < start or ls > stop:
                pos_idx_hat.append(ls)

        ancor_labels = self.minClass[start:stop] # 标签
        # 数据
        ancor_datas = []
        for j in ancor_labels:
            ancor_datas.append(self.train_data.data[j].view(self.dataShape).tolist())
        ancor_datas = torch.FloatTensor(ancor_datas)
        pos_datas = []
        neg_datas = []
        for k, label in enumerate(ancor_labels):
            neg_idx = np.where(self.majorClass)  # 负样本
            neg_idx = (self.majorClass)  # 负样本
            neg_datas.append( self.train_data.data[np.random.choice(neg_idx)].view(self.dataShape).tolist())
            # pos 样本
            #pos_idx = np.where(self.minClass)[0] # 直接寻找 y


            if len(pos_idx_hat):
                pos_datas.append( self.train_data.data[np.random.choice(pos_idx_hat)].view(self.dataShape).tolist())
            else:
                # positive examples are within the batch or just 1 example in dataset
                pos_datas.append( self.train_data.data[np.random.choice(pos_idx)].view(self.dataShape).tolist())
        neg_datas = torch.FloatTensor(neg_datas)
        pos_datas = torch.FloatTensor(pos_datas)

        # 对于MNIST 直接赋值
        ANCHOR = (ancor_datas)
        POSITIVE = (pos_datas)
        NEGATIVE = (neg_datas)
        return [ANCHOR, POSITIVE, NEGATIVE]



