import torch
import numpy as np
from torch import nn
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

train_version = 0
test_version = 0


class MLP(nn.Module):
    def __init__(self, layers = 2):
        super(MLP, self).__init__()
        
        # self.layers = layers
        # tmp = [(4 ** (layers - i)) for i in range(layers-1)]
        # self.in_dims = [768] + tmp
        # self.out_dims = tmp + [1]
        
        # self.net = [nn.Linear(self.in_dims[i], self.out_dims[i]) for i in range(self.layers)]
        # self.net = nn.Sequential(*self.net)
        
        self.fc1 = nn.Linear(768, 256)
        self.fc4 = nn.Linear(256, 1)
    
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        # for i in range(self.layers):
        #     if i == self.layers - 1:
        #         x = torch.sigmoid(self.net[i](x)) * 100 # for regrading
        #     else: 
        #         x = F.relu(self.net[i](x))
        return x


def get_info(is_train):
    if is_train:
        f = open('./src/train_set.txt')
    else:
        f = open('./src/test_set.txt')
    data = {}
    for line in f:
        if line != '':
            tmp = line.split(' ')
            tmp[1] = float(tmp[1].replace('\n', ''))
            data[tmp[0]] = tmp[1]
    f.close()
    return data

def read_data_train(version=1):
    x_train = []
    y_train = []
    # get the train data
    data = get_info(1)
    for item in data:
        if os.path.exists("./src/train_transform/version" + str(version) + "/" + item + ".txt"):
            file = open("./src/train_transform/version" + str(version) + "/" + item + ".txt")
            tmp = []
            for line in file:
                tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
                tmp = [float(i) for i in tmp]
            x_train.append(tmp)
            y_train.append(data[item])
    return x_train, y_train

def read_data_test(version=1):
    x_test = []
    y_test = []
    # get the test data
    data = get_info(0)
    for item in data:
        if os.path.exists("./src/test_transform/version" + str(version) + "/" + item + ".txt"):
            file = open("./src/test_transform/version" + str(version) + "/" + item + ".txt")
            tmp = []
            for line in file:
                tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
                tmp = [float(i) for i in tmp]
            x_test.append(tmp)
            y_test.append(data[item])
    return x_test, y_test

# save the model by version
def save_model(model, version):
    torch.save(model.state_dict(), "./model/version" + str(version) + ".pt")

if __name__ == "__main__":
    # print(1)
    X, Y = read_data_train(train_version)
    # init the MLP model
    model = MLP()
    # init the loss function
    criterion = nn.MSELoss(reduction="mean")
    # number of epochs to train the model
    n_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # training and testing
    for epoch in range(n_epochs):
        train_loss = 0.0
        for x in range(len(X)):
            inputs = torch.from_numpy(np.array(X[x])).float().to(device)
            gt = torch.from_numpy(np.array(Y[x])).float().to(device)
            optimizer.zero_grad()
            output = model(inputs)
            # print(output, torch.tensor(Y[x]).float())
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss = train_loss/len(X)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
        if train_loss < 1:
            break
    # X is a list of input file names
    # Y is the corresponding list of output scores
    X, Y = read_data_test(test_version)

    # print(len(X), len(Y))
    passn = 0
    total = len(X)*(len(X)-1)/2
    for x in range(len(X)):
        for y in range(x+1, len(X)):
            inputs1 = torch.from_numpy(np.array(X[x])).float().to(device)
            inputs2 = torch.from_numpy(np.array(X[y])).float().to(device)
            out1 = model(inputs1).cpu().detach().numpy()
            out2 = model(inputs2).cpu().detach().numpy()
            train_out = (out1 >= out2)
            test_out = (Y[x] >= Y[y])
            if ((train_out and test_out) or (not train_out and not test_out)):
                passn += 1

    print("Number of training " + str(n_epochs))
    print("Pass rate:" + str(passn/total))


    save_model(model, train_version)
    # save_model(model, 9)

    # print train_version and test_version
    print("Train version: " + str(train_version))
    print("Test version: " + str(test_version))