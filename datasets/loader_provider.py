# -*- coding = utf-8 -*-
# @Time: 2025/10/18 11:05
# @Author: wisehone
# @File: loader_provider.py
# @SoftWare: PyCharm
import sys
sys.path.append('..')
sys.path.append('.')

from datasets.data_factor import *
from torch.utils.data import dataset
from torch.utils.data import dataloader
from config import dataset2path

def get_data_loader(root_data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD'):
    assert dataset in dataset2path, 'dataset doesn\'t found'

    if root_data_path is None or root_data_path == '':
        root_data_path = dataset2path[dataset]

    if dataset == 'MSL':
        dataset = MSL_data_loader(root_data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset = PSM_data_loader(root_data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset = SMAP_data_loader(root_data_path, win_size, step, mode)
    elif dataset == 'SMD':
        dataset = SMD_data_loader(root_data_path, win_size, step, mode)
    elif dataset == 'SWAT':
        dataset = SWAT_data_loader(root_data_path, win_size, step, mode)
    # assert isinstance(dataset, dataloader.DataLoader), 'cant find a suitable dataloader, dataset is not DataLoader'

    shuffle = False if mode == 'train' else False
    # shuffle = False

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        drop_last=True
    )
    return dataset, loader


if __name__ == '__main__':
    dataset, loader = get_data_loader(None, 4, 100, 100, 'test', 'SWAT')
    sm = 0
    for i, (data, label) in enumerate(loader):
        print(data.shape, label.shape)
        sm += label.sum()
        print('sum: ', label.sum())
        # break
    print(sm)