import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from statistics import mean
import matplotlib.pyplot as plt

# Data class
class Data(Dataset):

    def __init__(self):
        xvals, yvals = make_classification(n_classes=2, n_samples=500, n_features=4, random_state=0)
        self.x = torch.from_numpy(xvals.astype(np.float32))
        self.y = torch.from_numpy(yvals.astype(np.float32).reshape(yvals.shape[0],1))
        self.len = self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

# Custom model class
class log_reg(nn.Module):

    def __init__(self, input_size, output_size):
        super(log_reg, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self,x):
        yhat = torch.sigmoid(self.linear(x)) # Sigmoid function for logistic regression
        return yhat

# Model with nn.Sequential()
def sequential_model(input_size, output_size):
    model = nn.Sequential(nn.Linear(input_size, output_size), nn.Sigmoid())
    return model

# Loading data in with class and dataloader object
data = Data()
loader = DataLoader(dataset = data, batch_size=10, shuffle=True)

input_size = data.x.shape[1]
output_size = 1

# Calling both models
model1 = log_reg(input_size, output_size)
model2 = sequential_model(input_size, output_size)

# Training loop
def train_model(model, learning_rate, epochs, plot_title, c):

    opt = optim.SGD(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    to_plot = []

    for epoch in range(epochs + 1):
        mean_loss_list = []
        for x,y in loader :

            opt.zero_grad()
            yhat = model.forward(x)

            loss = criterion(yhat, y)
            mean_loss_list.append(loss.item())
            loss.backward()
            
            opt.step()

        mean_loss = mean(mean_loss_list)
        to_plot.append(mean_loss)

        if epoch % (epochs/10) == 0 :
            print(f'Epoch {epoch} loss : {mean_loss}')

    ax[c].set_title(plot_title)
    ax[c].plot(to_plot)
    ax[c].annotate(f'{round(to_plot[-1], 3)}', (epoch, to_plot[-1]), size = 10)
    ax[c].grid(True)

learning_rate = 0.001
epochs = 200

# Setting up the fig and ax to draw loss graphs
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
fig.suptitle('Binary crossentropy loss', size = 20)

# Calling the train function on both models
print('nn.Module model : ')
train_model(model1, learning_rate, epochs, 'nn.Module model', 0)

print('-' * 50)

print('nn.Sequential model : ')
train_model(model2, learning_rate, epochs, 'nn.Sequential model', 1)

plt.show()