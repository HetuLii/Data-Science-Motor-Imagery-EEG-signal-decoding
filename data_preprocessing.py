from scipy import signal
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import torchvision.transforms as transforms
transf = transforms.ToTensor()
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import random

# This file provides data processing methods and the functions to make a cross-subject training and testing datasets

def buttferfiter(data):
    Fs = 250
    b, a = signal.butter(4, [0.1, 39], 'bandpass',fs=Fs)
    data = signal.filtfilt(b, a, data, axis=1)
    return data

class feature_dataset(Dataset):
    def __init__(self, file_paths, target_paths, transform=None):
        self.transform = transform
        # Initialize empty lists to store data and targets
        data_list = []
        target_list = []

        # Iterate over provided file paths to load and process each dataset
        for file_path, target_path in zip(file_paths, target_paths):
            data = self.parse_data_file(file_path)
            target = self.parse_target_file(target_path)
            data_list.append(data)
            target_list.append(target)

        # Concatenate all data and target arrays
        self.data = np.concatenate(data_list, axis=0)
        self.target = np.concatenate(target_list, axis=0)

    def parse_data_file(self, file_path):
        # Extract the predictors from dataset
        data = pd.read_csv(file_path, header=None)
        data = np.array(data, dtype=np.float32)
        data =  buttferfiter(data)
        num_sub = len(data)
        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        data = data.reshape(num_sub, 22, 1000)
        data = torch.tensor(data)
        return np.array(data, dtype=np.float32)

    def parse_target_file(self, target_path):
        # Extract the targets from dataset
        target = pd.read_csv(target_path, header=None)
        target = np.array(target, dtype=np.float32)
        encoder = OneHotEncoder(handle_unknown='ignore')
        x_hot = target.reshape(-1, 1)
        encoder.fit(x_hot)
        x_oh = encoder.transform(x_hot).toarray()
        d = torch.tensor(x_oh) 
        d = torch.squeeze(d)
        return np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index,:]
        #index2 = np.random.choice(len(self.data))
        #x2 = self.data[index2,:]
        target = self.target[index,:]
        return x,target
    
'''This function checks whether the subject number is valid. Change the index'''
def check_integer(x, num):
    try:
        # Check if x is an integer
        if not isinstance(x, int):
            raise ValueError("The value must be an integer.")
        
        # Check if x is between 1 and num (9 here)
        if not (1 <= x <= num): 
            raise ValueError("The value must be smaller than the maximum subject number.")
        
        print(f"{x} is a valid integer between 1 and {num}")
    
    except ValueError as e:
        print(f"Error: {e}")
    
def make_dataset(num_subjects, subject, base_data_path, base_target_path, data_file_pattern, target_file_pattern, batch_size):
    """
    Generate training and validation datasets. I design the function such that it's more convenient to replicate

    Parameters:
    - num_subjects: The total number of subjects.
    - subject: The subject whose dataset is the testing set.
    - base_data_path: The base path where the data files are stored.
    - base_target_path: The base path where the target files are stored.
    - data_file_pattern: A pattern to generate the names of the data files.
    - target_file_pattern: A pattern to generate the names of the target files.
    - batch_size: The number of samples in a batch
    
    Returns:
    - A tuple of (train_dataset, val_dataset).
    """

    # Randomly select one subject as the test subject
    x = subject 
    check_integer(x, num_subjects)

    # Generate file paths for the validation subject
    val_path = [
        f'{base_data_path}/{data_file_pattern.format(x)}E.csv', 
        f'{base_data_path}/{data_file_pattern.format(x)}T.csv'
    ]
    val_target_path = [
        f'{base_target_path}/{target_file_pattern.format(x)}E.csv',
        f'{base_target_path}/{target_file_pattern.format(x)}T.csv'
    ]
    val_dataset = feature_dataset(file_paths = val_path,
                                    target_paths = val_target_path)
    val_dataloader = DataLoader(val_dataset,shuffle=True,batch_size=batch_size)

    # Initialize lists for training paths
    train_path = []
    target_path = []

    # Populate training paths excluding the validation subject
    for i in range(1, num_subjects + 1):
        if i != x:
            train_path.append(f'{base_data_path}/{data_file_pattern.format(i)}E.csv')
            train_path.append(f'{base_data_path}/{data_file_pattern.format(i)}T.csv')
            target_path.append(f'{base_target_path}/{target_file_pattern.format(i)}E.csv')
            target_path.append(f'{base_target_path}/{target_file_pattern.format(i)}T.csv')

    train_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = feature_dataset(file_paths = train_path,
                                    target_paths = target_path)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

    return val_dataloader, train_dataloader

