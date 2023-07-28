from torch.utils.data import DataLoader
import numpy as np
import torch

def myProcessData(all_data, sequence_length, input_size, batch_size):
    # Data preprocessing -----------------------------------------------
    # dividing training dataset, validation dataset, test dataset 
    train_ratio = 0.8
    valid_ratio = 0.1
    train_size = int(len(all_data)*train_ratio)
    valid_size = int(len(all_data)*valid_ratio)
    train_dataset = all_data[:train_size]
    valid_dataset = all_data[train_size:train_size+valid_size]
    test_dataset = all_data[train_size+valid_size:]

    # Standardization for each dataset
    train_mean = np.mean(train_dataset)
    train_max = np.max(train_dataset)
    train_min = np.min(train_dataset)
    train_dataset = (train_dataset - train_mean) / (train_max - train_min)

    test_mean = np.mean(test_dataset)
    test_max = np.max(test_dataset)
    test_min = np.min(test_dataset)
    test_dataset = (test_dataset - test_mean) / (test_max - test_min)

    # Convert to sequence data
    train_sequences = []
    test_sequences = []
    gap = 1
    for i in range(0, len(train_dataset) - sequence_length + 1, gap):
        sequence = train_dataset[i:i+sequence_length]
        train_sequences.append(sequence)
    for i in range(0, len(test_dataset) - sequence_length + 1, gap):
        sequence = test_dataset[i:i+sequence_length]
        test_sequences.append(sequence)

    print(len(train_sequences))
    print(len(test_sequences))

    # for numpy array
    train_data = np.array(train_sequences)
    test_data = np.array(test_sequences)
    input_size = 1  # Dimension of input data
    # make 3D. -> pytorch LSTM needs 3D dataset
    limited_length = (len(train_data) // sequence_length) * sequence_length
    train_data = np.reshape(train_data[:limited_length], (-1, sequence_length, input_size))
    limited_length = (len(test_data) // sequence_length) * sequence_length
    test_data = np.reshape(test_data, (-1, sequence_length, input_size))
    # Tensor Type dataset, dataloader
    train_subset_dataset = torch.Tensor(train_data)
    train_dataloader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_subset_dataset = torch.Tensor(test_data)
    test_dataloader = DataLoader(test_subset_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader