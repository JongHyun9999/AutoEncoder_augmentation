from torch.utils.data import DataLoader
import numpy as np
import torch

def myProcessData(all_data, input_size, batch_size):
    # Data preprocessing -----------------------------------------------
    # dividing training dataset, validation dataset, test dataset 
    train_ratio = 0.8
    valid_ratio = 0.1
    train_dataset = []
    valid_dataset = []
    test_dataset = []
    train_size = int(len(all_data[0])*train_ratio)
    valid_size = int(len(all_data[0])*valid_ratio)

    for i in range(input_size):
        train_data = all_data[i][:train_size]
        train_dataset.append(train_data)
        valid_data = all_data[i][train_size:train_size+valid_size]
        test_data = all_data[i][train_size+valid_size:]
        test_dataset.append(test_data)

        # Standardization for each dataset
        # pjh. normalization을 통한 std domain으로 바꾸는건 의미가 없다고 판단.
        # 온도의 흐름을 담기 위해 standardization 도입했음.
        train_max = np.max(train_dataset[i])
        train_min = np.min(train_dataset[i])
        train_dataset[i] = (train_dataset[i] - train_min) / (train_max - train_min)

        test_max = np.max(test_dataset[i])
        test_min = np.min(test_dataset[i])
        test_dataset[i] = (test_dataset[i] - test_min) / (test_max - test_min)

    # for numpy array
    train_data = np.array(train_dataset)
    train_data = np.transpose(train_data)
    test_data = np.array(test_dataset)
    test_data = np.transpose(test_dataset)
    # make 3D. -> pytorch LSTM needs 3D dataset
    train_data = np.reshape(train_data, (-1, 1, input_size)) # (train_size, 1, input_size)
    test_data = np.reshape(test_data, (-1, 1, input_size))
    # Tensor Type dataset, dataloader
    train_subset_dataset = torch.Tensor(train_data)
    train_dataloader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_subset_dataset = torch.Tensor(test_data)
    test_dataloader = DataLoader(test_subset_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader