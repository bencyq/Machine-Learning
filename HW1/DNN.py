import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.pyplot import figure
from torch import nn
from torch.utils.data import Dataset, DataLoader
import csv

print(torch.cuda.is_available())
tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'
#########################
# 固定随机数，保证网络输出一致
#########################

myseed = 999  # 设置种子
torch.backends.cudnn.deterministic = True  # 固定cuda的随机数种子
torch.backends.cudnn.benchmark = False  # benchmark可以让cuda自动寻找合适的算法，但是这样每次网络的输出就不一样
np.random.seed(myseed)  # 设置numpy的种子
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


###############################
# Dataset 用于对数据集的切割和预处理
###############################


class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # 读取文件到ndarrays
        df = pd.read_csv(path)
        data = np.array(df.iloc[1:, 1:]).astype(np.float64)
        if not target_only:
            feats = range(93)
        else:
            pass
        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # 特征归一化
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)
        self.dim = self.data.shape[1]
        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, item):
        if self.mode in ['train', 'dev']:
            # 训练的时候一次性返回 x 和 y
            return self.data[item], self.target[item]
        else:
            # 测试的时候只返回 x
            self.data[item]

    def __len__(self):
        return len(self.data)


############################################
# DataLoader 用于把 Dataset里的数据放入batches中
############################################


def pre_data_loader(path, mode, batch_size, n_jobs=0, target_only=False):
    # 先构建 Dataset
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    # 再构建 DataLoader
    data_loader = DataLoader(dataset, batch_size, shuffle=mode == 'train', drop_last=False, num_workers=n_jobs,
                             pin_memory=True)  # 当mode为train时候，shuffle为True，即读入数据的时候不按照顺序
    return data_loader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        # 继承父类的init
        super(NeuralNet, self).__init__()
        # 定义自己的网络结构
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 定义loss_fn
        self.criterion = nn.MSELoss(reduction='mean')
        """
        reduction='none' 返回向量
        reduction='mean' 返回平均值--默认情况下
        reduction='sum'  返回求和值
        """

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss = total_loss / len(dv_set.dataset)
    return total_loss


def  test(tt_set, model, device):
    model.eval()  # set model to evaluation mode
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()  # 把所有数据拼接为一行，并且转换为 ndarray
    return preds


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # 把模型设置为 training 模式
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('保存模型 epoch = {:4d}, loss = {:.4f}'.format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:  # dev_loss > min_mse 多次后终止运行，保证模型不会进行无效的过拟合运算
            break
    print('运行结束  epoch = {}'.format(epoch))
    return min_mse, loss_record


if __name__ == '__main__':
    device = get_device()
    os.makedirs('models', exist_ok=True)
    target_only = False
    config = {
        'n_epochs': 3000,
        'batch_size': 270,
        'optimizer': 'SGD',
        'optim_hparas': {
            'lr': 0.001,
            'momentum': 0.9  # 动量
        },
        'early_stop': 200,
        'save_path': 'models./model.pth'
    }
    tr_set = pre_data_loader(tr_path, 'train', config['batch_size'], target_only=target_only)
    dv_set = pre_data_loader(tr_path, 'dev', config['batch_size'], target_only=target_only)
    tt_set = pre_data_loader(tt_path, 'test', config['batch_size'], target_only=target_only)
    model = NeuralNet(tr_set.dataset.dim).to(device)
    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

    ##########
    # 结果可视化
    ##########
    plot_learning_curve(model_loss_record, title='deep model')
    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    plot_pred(dv_set, model, device)  # Show prediction on the validation set


    def save_pred(preds, file):
        ''' Save predictions to specified file '''
        print('Saving results to {}'.format(file))
        with open(file, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])


    ############
    # 保存预测结果
    ############
    preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
    save_pred(preds, 'pred.csv')  # save prediction file to pred.csv
