import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pandas as pd
import random


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_met, window=24, horizon=24, dtype=torch.float):
        super().__init__()
        self._data = data['cloud']
        self.ground = data_met['ground']
        self.sky = data_met['sky']

 
        self.cloud_list = np.stack(list(self._data.keys())).astype(int)
        self.cloud_all = np.stack(list(self._data.values()))
        self.met_list = np.stack(list(self.ground.keys())).astype(int)
        
        self._window = window
        self._horizon = horizon
        self._dtype = dtype
      

    def __getitem__(self, index):
        index = np.minimum(len(self._data)-self._window-self._horizon-1, index)

        #met_ind = np.where(self.met_list<=self.cloud_list[index+self._window+1])[0][-1]
        met_ind = find_nearest(self.met_list, self.cloud_list[index+self._window])
        
        cloud_x = self.cloud_all[index:index+self._window]
        cloud_x= cloud_x/255
        
        ground_x = self.ground[str(met_ind)]
        sky_x = self.sky[str(met_ind)]
        
        cloud_y = self.cloud_all[index+self._window+5:index+self._window+self._horizon:6]
        cloud_y = cloud_y/255
        
        x = np.concatenate([cloud_x,ground_x,sky_x],axis=0)
            
        x = torch.from_numpy(x).type(self._dtype)
        cloud_y = torch.from_numpy(cloud_y).type(self._dtype)
        
        return x, cloud_y
    
    
    def __len__(self):
        return len(self._data)-self._window-self._horizon-1
    
    


def generate_valid_set(Cfg):
    valid_fold_list = np.random.choice(50, Cfg.batch_size)#random.sample(range(0, 50), Cfg.batch_size)
    data = pd.read_pickle('data/data_test.pkl')
    data_met = pd.read_pickle('data/data_test_met.pkl')
    data = data['cloud']   
    ground = data_met['ground']
    sky = data_met['sky']

    cloud_clip = np.load("data/valid_cloud_clip.npy",allow_pickle=True)
    met_clip = np.load('data/test_meta_clip.npy',allow_pickle=True)

    X = []
    Y = []
    for valid_id in valid_fold_list:
        clip = np.sort(cloud_clip[valid_id])
        start = random.sample(range(0, 47), 1)[0]
        end = start+24
        train_ind = clip[start:end]
        test_ind = clip[end+5:end+24:6]
        
        met_ind = find_nearest(met_clip[valid_id], clip[end])

        x = []
        y = []
        for i in train_ind:
            x.append(data[str(i)])
        for j in test_ind:
            y.append(data[str(j)])
        x = np.stack(x)/255
        y = np.stack(y)/255
        
        x = np.concatenate([x, ground[str(met_ind)], sky[str(met_ind)] ],axis=0)
        
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    X = torch.from_numpy(X).type(torch.float)
    Y = torch.from_numpy(Y).type(torch.float)
    
    return X, Y