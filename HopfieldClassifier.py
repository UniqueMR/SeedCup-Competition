import matplotlib.pyplot as plt
import os
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import _parse_input_dimensions
import pandas
import torch
import numpy
from sklearn.model_selection import KFold
from torch._C import device
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear

from modules import Hopfield, HopfieldPooling, HopfieldLayer

def dataloader(path):
    data = pandas.read_csv(path)
    data = numpy.array(data)
    length = len(data)
    dataset = []
    datatag = []
    for i in range(length):
        dataset.append(data[i][0:32])
        datatag.append(data[i][32])
    return dataset, datatag

def split(dataset, datatag, nfolds):
    data_num = len(dataset)
    kf = KFold(nfolds, shuffle=True, random_state=None)
    datasets = []
    for train_idx, test_idx in kf.split(dataset):
        data = {}
        data['train_data'] = [dataset[i] for i in train_idx]
        data['train_tag'] = [datatag[i] for i in train_idx]
        data['test_data'] = [dataset[i] for i in test_idx]
        data['test_tag'] = [datatag[i] for i in test_idx]
        datasets.append(data)
    return datasets

def predict(y_pred):
    if y_pred[0][0] > y_pred[0][1]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    path = 'E:\\seedCup Competition\\真测试集SVM.data'
    test = numpy.loadtxt(path, dtype=float, delimiter=',')
    dataset, datatag = dataloader("E:\\seedCup Competition\\均衡.csv")
    nfolds = 30
    datasets = split(dataset,datatag,nfolds)
    learning_rate = 1e-3
    num_epochs = 50
    RunDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(RunDevice)
    del dataset
    del datatag
    num_characters = len(datasets[0]['train_data'][0])
    hopfield = Hopfield(
        input_size=num_characters,
        hidden_size=8,
        num_heads=8,
        update_steps_max = 3,
        scaling=0.25,
        dropout=0.5
    )
    print(hopfield.output_size * num_characters)
    output_projection = torch.nn.Linear(in_features=num_characters, out_features=2)
    network = torch.nn.Sequential(hopfield, torch.nn.Flatten(), output_projection, torch.nn.Flatten(start_dim=0)).to(device = RunDevice)
    optimizer = torch.optim.AdamW(params=network.parameters(), lr=learning_rate)
    lossFunc = torch.nn.CrossEntropyLoss()

    i = 0

    accuracies = []
    for fold in range(nfolds):
        i = 0
        dataset = datasets[fold]
        train_datas = dataset['train_data']
        train_tags = dataset['train_tag']
        test_datas = dataset['test_data']
        test_tags = dataset['test_tag']
        losses = []
        for epoch in range(num_epochs):
            # training process
            network.train()
            i = 0
            print('training')
            for train_data in train_datas:
                train_data = torch.tensor(train_data)
                train_data = torch.unsqueeze(train_data,0).to(torch.float32)
                train_data = torch.unsqueeze(train_data,0).to(device = RunDevice)
                y_pred = network.forward(input = train_data)
                y_pred = torch.unsqueeze(y_pred,0)
                y = torch.tensor([train_tags[i]]).to(device = RunDevice)
                y = y.to(torch.long)
                i += 1
                loss = lossFunc(y_pred, y)
                if i % 100 == 0:
                    # print(loss)
                    losses.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # plt.figure(fold)
        # plt.plot(losses)
        # plt.ylabel('loss')
        # plt.show()

        i = 0
        right = 0
        print('local testing')
        for test_data in test_datas:
            test_data = torch.tensor(test_data)
            test_data = torch.unsqueeze(test_data,0).to(torch.float32)
            test_data = torch.unsqueeze(test_data,0).to(device=RunDevice)
            y_pred = network.forward(input=test_data)
            y_pred = torch.unsqueeze(y_pred,0)
            pred = predict(y_pred)
            y = test_tags[i]
            i += 1
            if pred == y:
                right += 1
        accuracy = right / i
        print(accuracy)
        accuracies.append(accuracy)

        # path = 'E:\\seedCup Competition\\真测试集SVM.data'
        # test = numpy.loadtxt(path, dtype=float, delimiter=',')
        print('online testing')
        for sample in test:
            sample = torch.tensor(sample)
            sample = torch.unsqueeze(sample,0).to(torch.float32)
            sample = torch.unsqueeze(sample,0).to(device=RunDevice)
            y_pred = network.forward(input=sample)
            y_pred = torch.unsqueeze(y_pred,0)
            pred = predict(y_pred)
            print(pred)
        
    # plt.figure(1)
    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.show()



    # network.train()

    # train_data = torch.tensor(datasets[0]['train_data'][0])
    # print(train_data)
    # train_data = torch.unsqueeze(train_data,0).to(torch.float32)
    # train_data = torch.unsqueeze(train_data,0).to(device = RunDevice)
    # y_pred = network.forward(input = train_data)
    # y_pred = torch.unsqueeze(y_pred,0)
    # print(y_pred)
    # y = torch.tensor([datasets[0]['train_tag'][0]]).to(device = RunDevice)
    # y = y.to(torch.long)
    # loss = lossFunc(y_pred, y)
    # print(loss)
