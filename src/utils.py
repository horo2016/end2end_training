"""
Helper functions.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import os
import pandas as pd

from torch.utils import data
from RcCarDataset import TripletDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean_std = ([0.406], [0.225]) # mean-std
img_transform = transforms.Compose([transforms.ToTensor(), 
                                   # transforms.Normalize(*mean_std)
                                   ])

def toDevice(datas,target, device):
    """
    Enable cuda.

    Args:
        datas: tensor
        device: cpu or cuda

    Returns:
        Transform `data` to `device`
    """

    return datas.float().to(device), target.float().to(device)


def load_data(data_dir, train_size):
    """
    Load training data and train-validation split.

    Args:
        data_dir: data root
        train_size: ratio to split to training set and validation set.

    Returns:
        trainset: training set
        valset: validation set
    """
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'),
                          names=[ 'throttle', 'steering','center'])
    #print(data_df)
    # Divide the data into training set and validation set
    train_len = int(train_size * data_df.shape[0])
    valid_len = data_df.shape[0] - train_len
    trainset, valset = data.random_split(
        data_df.values.tolist(),
        lengths=[train_len, valid_len]
    )
    print(str(train_len) + ":"+ str(valid_len ))
    return trainset, valset


def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers):
    """Self-Driving vehicles simulator dataset Loader.

    Args:
        trainset: training set
        valset: validation set
        batch_size: training set input batch size
        shuffle: whether shuffle during training process
        num_workers: number of workers in DataLoader

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    '''transformations = transforms.Compose(
        [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])
    '''
    # Load training data and validation data
    training_set = TripletDataset(dataroot, trainset, img_transform)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    validation_set = TripletDataset(dataroot, valset, img_transform)
    valloader = DataLoader(validation_set,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

    return trainloader, valloader
