import torch
import numpy as np
from torch import nn
import os
import csv

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import sys, getopt

TRAIN_VERSION = 3
TEST_VERSION = 3
MODEL_PATH = "../src/model/version3/3_324.pth"
INPUT_PATH = "./hand_transform/"
METHOD = "MLP"
PWC = True
CN2 = False


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # define the layers of the neural network
        self.fc1 = nn.Linear(768, 512)
        self.fc1_drop = nn.Dropout(p=0.2)
        self.fc1_regularization = nn.L1Loss()
        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc2_regularization = nn.L1Loss()
        self.fc3 = nn.Linear(256, 128)
        self.fc3_drop = nn.Dropout(p=0.2)
        self.fc3_regularization = nn.L1Loss()
        self.fc4 = nn.Linear(128, 1)
        self.fc4_regularization = nn.L1Loss()

    def forward(self, x):
        # define the forward pass of the neural network
        x = self.fc1(x)
        x = self.fc1_drop(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc2_drop(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.fc3_drop(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = torch.sigmoid(x) * 100

        return x
    
# class MLP(nn.Module):
#     def __init__(self, layers = 2):
#         super(MLP, self).__init__()
        
#         # self.layers = layers
#         # tmp = [(4 ** (layers - i)) for i in range(layers-1)]
#         # self.in_dims = [768] + tmp
#         # self.out_dims = tmp + [1]
        
#         # self.net = [nn.Linear(self.in_dims[i], self.out_dims[i]) for i in range(self.layers)]
#         # self.net = nn.Sequential(*self.net)
        
#         self.fc1 = nn.Linear(768, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 1)
    
    
#     def encode(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc4(x)
#         return x
    
#     def forward(self, x):
#         x = self.encode(x)
#         # for i in range(self.layers):
#         #     if i == self.layers - 1:
#         #         x = torch.sigmoid(self.net[i](x)) * 100 # for regrading
#         #     else: 
#         #         x = F.relu(self.net[i](x))
#         return x
    
# class MLP(nn.Module):
#     def __init__(self, layers = 2):
#         super(MLP, self).__init__()
        
#         # self.layers = layers
#         # tmp = [(4 ** (layers - i)) for i in range(layers-1)]
#         # self.in_dims = [768] + tmp
#         # self.out_dims = tmp + [1]
        
#         # self.net = [nn.Linear(self.in_dims[i], self.out_dims[i]) for i in range(self.layers)]
#         # self.net = nn.Sequential(*self.net)
        
#         self.fc1 = nn.Linear(768, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 1)
    
    
#     def encode(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc4(x)) * 100
#         return x
    
#     def forward(self, x):
#         x = self.encode(x)
#         # for i in range(self.layers):
#         #     if i == self.layers - 1:
#         #         x = torch.sigmoid(self.net[i](x)) * 100 # for regrading
#         #     else: 
#         #         x = F.relu(self.net[i](x))
#         return x

class CNN(nn.Module):
    def __init__(self, layers = 2):
        super(CNN, self).__init__()
        
        # input 1*64*768 shape
        
        # first layer
        # mid_out1 : 1*16*192
        self.conv1 = nn.Conv2d(1, 1, 4, stride=4)
        # mid_out2 : 1*1*12
        self.conv2 = nn.Conv2d(1, 1, 2, stride=2)
        
        # convert 1*1*12 to 1*1*1
        self.fc1 = nn.Linear(12, 1)
        

    
    def encode(self, x):
        x = self.conv1(F.relu(self.conv1(x)))
        x = self.conv2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        return x

def get_info(is_train):
    # for file in ./data, get the name of the file
    data=os.listdir(INPUT_PATH)
    return data

def read_data_train(version=1):
    x_train = []
    y_train = []
    # get the train data
    data = get_info(1)
    for item in data:
        if os.path.exists(INPUT_PATH + item ):
            file = open(INPUT_PATH + item )
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
        if os.path.exists(INPUT_PATH + item ):
            file = open(INPUT_PATH + item )
            tmp = []
            for line in file:
                tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
                tmp = [float(i) for i in tmp]
            x_test.append(tmp)
            # y_test.append(data[item])
    return x_test, y_test

# calc test loss
def test_loss(model, X, Y, METHOD):
    loss = 0
    sum = 0
    if METHOD == "MLP":
        for i in range(len(X)):
            out = model(torch.tensor(X[i]).float())
            if out > 100.0:
                out = 100.0
            loss += (out - Y[i]) ** 2
            sum += 1
    else:
        for i in range(len(X)):
            if len(X[i]) != 64*768:
                continue
            out = torch.from_numpy(np.array(X[i]).reshape(1, 64, 768)).float().to(device)
            out = model(out).cpu().detach().numpy()
            if out > 100.0:
                out = 100.0
            loss += (out - Y[i]) ** 2
            sum += 1
            # print("pred :" + str(out) + "; socre : " + str(Y[i]))
    return float(loss / sum)

if __name__ == "__main__":
    
    # If there are input parameters, let MODEL_PATH be the first parameter
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
        TEST_VERSION = int(sys.argv[2])
        METHOD = sys.argv[3]
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # init the MLP model
    if METHOD == "MLP":
        model = MLP()
    else:
        model = CNN()
    
    model.to(device)
    data = get_info(0)
    
    # load the model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    # init the loss function
    X, Y = read_data_test(TEST_VERSION)
    
    dict = {}
    
    # x_test = []
    # # get the test data
    # file = open( "covid_filter_out.a3m" )
    # tmp = []
    # for line in file:
    #     tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
    #     tmp = [float(i) for i in tmp]
    # x_test.append(tmp)
    # # print(x_test)
    # out = model(torch.tensor(x_test).float())
    # print(out)

    # pair-wise comparison
    if PWC:
        passn = 0
        total = 0
        x = 0
        if METHOD == "MLP":
            while x < len(X):
                # y=x+1
                out1 = model(torch.tensor(X[x]).float())
                # out2 = model(torch.tensor(X[y]).float())
                # print(data[x] + ": " + str(float(out1)))
                # print(data[y] + ": " + str(float(out2)))
                
                dict[data[x]] = float(out1)
                # dict[data[y]] = float(out2)
                      
                # train_out = (out1 >= out2)
                # test_out = (Y[x] >= Y[y])
                # if ((train_out and test_out) or (not train_out and not test_out)):
                #     passn += 1
                # total += 1
                x += 1
        else:
            while x < len(X):
                y = x + 1
                if len(X[x]) != 64*768 or len(X[y]) != 64*768:
                    x += 2
                    continue
                inputs1 = torch.from_numpy(np.array(X[x]).reshape(1, 64, 768)).float().to(device)
                inputs2 = torch.from_numpy(np.array(X[y]).reshape(1, 64, 768)).float().to(device)
                out1 = model(inputs1).cpu().detach().numpy()
                out2 = model(inputs2).cpu().detach().numpy()
                train_out = (out1 >= out2)
                test_out = (Y[x] >= Y[y])
                if ((train_out and test_out) or (not train_out and not test_out)):
                    passn += 1
                total += 1
                x += 2
        # print("Pair-wise comparison pass rate:" + str(passn/total))
                
        # print(dict)
        # sort for dict
        dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        print(dict)
        # output csv
        with open('result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id', 'score'])
            for item in dict:
                writer.writerow([item[0], item[1]])
    # C_N^2
    # if CN2:
    #     passn = 0
    #     total = 0
    #     if METHOD == "MLP":
    #         for x in range(len(X)):
    #             for y in range(x+1, len(X)):
    #                 out1 = model(torch.tensor(X[x]).float())
    #                 out2 = model(torch.tensor(X[y]).float())
    #                 train_out = (out1 >= out2)
    #                 test_out = (Y[x] >= Y[y])
    #                 if ((train_out and test_out) or (not train_out and not test_out)):
    #                     passn += 1
    #                 total += 1
    #     else:
    #         for x in range(len(X)):
    #             for y in range(x+1, len(X)):
    #                 if len(X[x]) != 64*768 or len(X[y]) != 64*768:
    #                     continue
    #                 inputs1 = torch.from_numpy(np.array(X[x]).reshape(1, 64, 768)).float().to(device)
    #                 inputs2 = torch.from_numpy(np.array(X[y]).reshape(1, 64, 768)).float().to(device)
    #                 out1 = model(inputs1).cpu().detach().numpy()
    #                 out2 = model(inputs2).cpu().detach().numpy()
    #                 train_out = (out1 >= out2)
    #                 test_out = (Y[x] >= Y[y])
    #                 if ((train_out and test_out) or (not train_out and not test_out)):
    #                     passn += 1
    #                 total += 1
    #     print("C_N^2 pass rate:" + str(passn/total))

    # test loss
    # print("Test loss:" + str(test_loss(model, X, Y, METHOD)))
