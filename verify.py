import torch
import numpy as np
from torch import nn
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

train_version = 3
test_version = 3


class MLP(nn.Module):
    def __init__(self, layers = 2):
        super(MLP, self).__init__()
        
        # self.layers = layers
        # tmp = [(4 ** (layers - i)) for i in range(layers-1)]
        # self.in_dims = [768] + tmp
        # self.out_dims = tmp + [1]
        
        # self.net = [nn.Linear(self.in_dims[i], self.out_dims[i]) for i in range(self.layers)]
        # self.net = nn.Sequential(*self.net)
        
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
    
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

# calc test loss
def test_loss(model, X, Y):
    loss = 0
    for i in range(len(X)):
        out = model(torch.tensor(X[i]).float())
        loss += (out - Y[i]) ** 2
    return loss / len(X)

if __name__ == "__main__":
    # init the MLP model
    model = MLP()
    # load the model
    model.load_state_dict(torch.load("./src/model/version3_1_9.pth"))
    # init the loss function
    X, Y = read_data_test(test_version)

    # pair-wise comparison
    passn = 0
    total = 0
    x = 0
    while x < len(X):
        y=x+1
        out1 = model(torch.tensor(X[x]).float())
        out2 = model(torch.tensor(X[y]).float())
        train_out = (out1 >= out2)
        test_out = (Y[x] >= Y[y])
        if ((train_out and test_out) or (not train_out and not test_out)):
            passn += 1
        total += 1
        x += 2
    print("Pair-wise comparison pass rate:" + str(passn/total))
                
    # C_N^2
    passn = 0
    total = 0
    for x in range(len(X)):
        for y in range(x+1, len(X)):
            out1 = model(torch.tensor(X[x]).float())
            out2 = model(torch.tensor(X[y]).float())
            train_out = (out1 >= out2)
            test_out = (Y[x] >= Y[y])
            if ((train_out and test_out) or (not train_out and not test_out)):
                passn += 1
            total += 1
    print("C_N^2 pass rate:" + str(passn/total))

    # test loss
    print("Test loss:" + str(test_loss(model, X, Y)))
