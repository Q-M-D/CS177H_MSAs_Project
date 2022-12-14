import torch
import numpy as np
from torch import nn
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 500
LEARNING_RATE = 0.0001
train_version = 3
test_version = 3
attempt = 3


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
def save_model(model, version, attempt, id):
    torch.save(model.state_dict(), "./src/model/version" + str(version) + "/" + str(attempt)+ "_" + str(id) + ".pth")
    
def test(X, Y, id):
    # print(len(X), len(Y))
    X, Y = read_data_test(test_version)
    passn = 0
    total = 0
    # for x in range(len(X)):
    #     for y in range(x+1, len(X)):
    x = 0
    while x < len(X):
        y = x + 1
        inputs1 = torch.from_numpy(np.array(X[x])).float().to(device)
        inputs2 = torch.from_numpy(np.array(X[y])).float().to(device)
        out1 = model(inputs1).cpu().detach().numpy()
        out2 = model(inputs2).cpu().detach().numpy()
        train_out = (out1 >= out2)
        test_out = (Y[x] >= Y[y])
        if ((train_out and test_out) or (not train_out and not test_out)):
            passn += 1
        # elif(epoch >=3):
        #     # print wrong test case and its score
        #     print("Expect out score " + str(x) +":" + str(Y[x]) + " " + str(y) + ":" + str(Y[y]))
        #     print("Wrong out score: " + str(out1) + " " + str(out2))
        x += 2
        total += 1
        
    loss = 0.0
    for i in range(len(X)):
        inputs = torch.from_numpy(np.array(X[i])).float().to(device)
        outputs = model(inputs)
        loss += criterion(outputs, torch.from_numpy(np.array(Y[i])).float().to(device))
    loss = loss / len(X)
    print("Number of test " + str(id))
    print("Loss: " + str(loss))

    print("Number of training " + str(id))
    print("Pass rate:" + str(passn/total))


    save_model(model, train_version,attempt, id)
    # save_model(model, 9)

    # print train_version and test_version
    print("Train version: " + str(train_version))
    print("Test version: " + str(test_version))

if __name__ == "__main__":
    # print(1)
    X, Y = read_data_train(train_version)
    # init the MLP model
    model = MLP()
    # init the loss function
    criterion = nn.MSELoss(reduction="mean")
    # number of epochs to train the model
    n_epochs = EPOCH
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load the model
    model.load_state_dict(torch.load("./src/model/version" + str(train_version) + "/" + str(attempt) + "_400.pth"))
    model.to(device)
    
    # training and testing
    for epoch in range(n_epochs+1):
        if epoch == 0:
            epoch = 401
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
        # if epoch % 2== 0:
        test(X, Y, epoch)
        if train_loss < 0.0001:
            break
    # X is a list of input file names
    # Y is the corresponding list of output scores
    X, Y = read_data_test(test_version)

