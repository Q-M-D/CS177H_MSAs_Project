import torch
import numpy as np
from torch import nn
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 400
LEARNING_RATE = 0.001
train_version = 5
test_version = 5
attempt = 3


# create a CNN model
# input is a 1*64*768 matrix
# mid_out1 : 1*16*192
# mid_out2 : 1*1*12
# output is a 1*1*1 matrix
# two layers of CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # input 1*64*768 shape

        # first layer
        # mid_out1 : 1*16*192
        self.conv1 = nn.Conv2d(1, 16, 4, stride=4)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        # second layer
        # mid_out2 : 1*32*48
        self.conv2 = nn.Conv2d(16, 32, 4, stride=4)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # third layer
        # mid_out3 : 1*64*12
        self.conv3 = nn.Conv2d(32, 1, 1, stride=1)
        self.pool3 = nn.MaxPool2d(1, stride=1)

        # convert 1*64*12 to 1*1*1
        self.fc1 = nn.Linear(12, 1)
        self.fc_drop = nn.Dropout(p=0.2)
        

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.fc_drop(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.fc_drop(x)
        # print(x.shape)
        x = self.pool3(F.relu(self.fc_drop(self.conv3(x))))
        # x = self.fc_drop(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
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
    print(version)
    # get the train data
    data = get_info(1)
    for item in data:
        if os.path.exists("./src/train_transform/version" + str(version) + "/" + item + ".txt"):
            file = open("./src/train_transform/version" + str(version) + "/" + item + ".txt")
            tmp = []
            # print(item)
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
            # print(item)
            for line in file:
                tmp = line.replace('[', '').replace(']', '').replace(' ', '').split(',')
                tmp = [float(i) for i in tmp]
            x_test.append(tmp)
            y_test.append(data[item])
    return x_test, y_test

# save the model by version
def save_model(model, version, attempt, id):
    torch.save(model.state_dict(), "./src/model/version" + str(version) + "/" + str(attempt)+ "_" + str(id) + ".pth")
    
def test(X, Y, id, model):
    # print(len(X), len(Y))
    # X, Y = read_data_test(test_version)
    passn = 0
    total = 0
    loss = 0.0
    # for x in range(len(X)):
    #     for y in range(x+1, len(X)):
    x = 0
    while x < len(X):
        y = x + 1
        if len(X[x]) != 64*768 or len(X[y]) != 64*768:
            x += 2
            continue
        inputs1 = torch.from_numpy(np.array(X[x]).reshape(1, 1, 64, 768)).float().to(device)
        inputs2 = torch.from_numpy(np.array(X[y]).reshape(1, 1, 64, 768)).float().to(device)
        out1 = model(inputs1).cpu().detach().numpy()
        out2 = model(inputs2).cpu().detach().numpy()
        train_out = (out1 >= out2)
        test_out = (Y[x] >= Y[y])
        loss += (out1 - Y[x]) ** 2
        loss += (out2 - Y[y]) ** 2
        if ((train_out and test_out) or (not train_out and not test_out)):
            passn += 1
        # elif(epoch >=3):
        #     # print wrong test case and its score
        #     print("Expect out score " + str(x) +":" + str(Y[x]) + " " + str(y) + ":" + str(Y[y]))
        #     print("Wrong out score: " + str(out1) + " " + str(out2))
        x += 2
        total += 1

    print("Number of training " + str(id))
    print("Pass rate:" + str(passn/total))
    print("Loss : " + str(loss/(total*2)))


    save_model(model, train_version,attempt, id)
    # save_model(model, 9)

    # print train_version and test_version
    print("Train version: " + str(train_version))
    print("Test version: " + str(test_version))

if __name__ == "__main__":
    # print(1)
    print("main in")
    X2, Y2 = read_data_test(test_version)
    X1, Y1 = read_data_train(train_version)
    print("read data done")
    # init the MLP model
    model = CNN()
    # init the loss function
    criterion = nn.MSELoss(reduction="mean")
    # number of epochs to train the model
    n_epochs = EPOCH
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # training and testing
    for epoch in range(n_epochs+1):
        train_loss = 0.0
        for x in range(len(X1)):
            # if size of X1 is not 49152, then skip it
            if len(X1[x]) != 49152:
                continue
            inputs = torch.from_numpy(np.array(X1[x]).reshape(1, 1, 64, 768)).float().to(device)
            gt = torch.from_numpy(np.array(Y1[x])).float().to(device)
            optimizer.zero_grad()
            output = model(inputs).float()
            # print(type(output), type(gt))
            # print(output, torch.tensor(Y[x]).float())
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss = train_loss/len(X1)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
        # if epoch % 2== 0:
        test(X2, Y2, epoch, model)
        if train_loss < 0.0001:
            break
    # X is a list of input file names
    # Y is the corresponding list of output scores

