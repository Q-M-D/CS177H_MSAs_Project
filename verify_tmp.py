import torch
import numpy as np
from torch import nn
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

train_version = 0


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 16)
        self.fc4 = nn.Linear(16, 1)
    
    
    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        return x


def get_info(is_train):
    if is_train:
        f = open('./train_set.txt')
    else:
        f = open('./test_set.txt')
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
        if os.path.exists("./train_transform/version" + str(version) + "/" + item + ".txt"):
            file = open("./train_transform/version" + str(version) + "/" + item + ".txt")
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
        if os.path.exists("./test_transform/version" + str(version) + "/" + item + ".txt"):
            file = open("./test_transform/version" + str(version) + "/" + item + ".txt")
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


# init the MLP model
model = MLP()
# load the model
model.load_state_dict(torch.load("./model/version" + str(train_version) + ".pt"))
# init the loss function
X, Y = read_data_train(train_version)

# print(len(X), len(Y))
passn = 0
total = len(X)*(len(X)-1)/2
for x in range(len(X)):
    for y in range(x+1, len(X)):
        out1 = model(torch.tensor(X[x]).float())
        out2 = model(torch.tensor(X[y]).float())
        train_out = (out1 >= out2)
        test_out = (Y[x] >= Y[y])
        if ((train_out and test_out) or (not train_out and not test_out)):
            passn += 1

print("Pass rate:" + str(passn/total))
