import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):

    def __init__(self):
        pass
    
    def __getitem__(self,index):
        pass

    def __len__(self):
        pass


class log_reg(nn.Module):

    def __init__(self, input_size, output_size):
        super(log_reg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self,x):
        yhat = torch.sigmoid(self.linear(x)) # Sigmoid function for logistic regression
        return yhat

input_size = 
output_size = 1
model1 = log_reg(input_size, output_size)
model2 = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())

    