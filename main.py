import sys
import shutil
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop, RandomRotation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from models.lenet import LeNet


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def plot_loss(df):
    g = sns.lmplot(
        x='step',
        y='loss_scores',
        data=df,
        col='model_n',
        hue='model_n',
        col_wrap=2,
        fit_reg=False
    ).set_axis_labels("Steps", "Loss")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Loss")
    plt.show()


def plot_accuracy(df):
    g = sns.lmplot(
        x='step',
        y='acc_scores',
        data=df,
        col='model_n',
        hue='model_n',
        col_wrap=2,
        fit_reg=False
    ).set_axis_labels("Steps", "Accuracy")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Model Accuracy")
    plt.show()


def unpack_data(data):
    iterations = [i[0] for i in data]
    loss_scores = [i[1] for i in data]
    acc_scores = [i[2] for i in data]

    return iterations, loss_scores, acc_scores


def load_data(training=True):
    transform_ = transforms.Compose(
        [RandomRotation(45),
         RandomCrop(28),
         transforms.ToTensor()]
    )
    data = torchvision.datasets.MNIST(
        root='./data/',
        train=training,
        download=True,
        transform=transform_,
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )

    return loader


def train(model, dataloader, criterion, optimizer, verbose=False):
    scores = []
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        # wrap features as torch Variables
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels)  # optimization
        loss.backward()  # compute back propagation
        optimizer.step()  # update model parameters

        running_loss += loss.data[0]

        if i % 100 == 99:  # print every 2000 mini-batches
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            accuracy = 100. * correct / total

            scores.append((i+1, running_loss/100, accuracy))

            # print results
            if verbose and i % 500 == 499:
                print("Batch: %5d - Loss: %.3f" % (i+1, running_loss/100))
                print("Accuracy: {:.2f}%".format(accuracy))

            running_loss = 0.0
    print()

    return scores


def main(models, training=True, verbose=True):
    results = defaultdict(list)
    for model in range(models):  # Our Epochs
        print("Model {}...".format(model+1))
        net = LeNet()
        print(net)
        dataloader = load_data(training=training)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters())
        scores = train(
            net,  # the model
            dataloader,  # the data provider
            criterion,  # the loss function
            optimizer,  # the optimization algorithm
            verbose=verbose,  # print results
        )
        step, loss, acc = unpack_data(scores)

        # reset gradients
        net.zero_grad()
        optimizer.zero_grad()

        # add observations ot the dictionary
        results['step'] += step
        results['loss_scores'] += loss
        results['acc_scores'] += acc
        results['model_n'] += [model+1] * len(step)

    if training:
        save_checkpoint({
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        })

    del net, criterion, optimizer

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Implementation of LeNet.",  # program title
    )
    parser.add_argument(
        '-models',  # argument name
        type=int,  # default data type
        help="The number of models to run",  # cli help description
        required=False,
        default=1,
    )
    parser.add_argument(
        '-training',
        type=bool,
        help='training?',
        required=False,
        default=True,
    )
    parser.add_argument(
        '-verbose',
        type=bool,
        help='print modeling training progress',
        required=False,
        default=False,
    )
    parser.add_argument(
        '-plot',
        type=bool,
        help='plot model results',
        required=False,
        default=False,
    )
    try:
        args = parser.parse_args(sys.argv[1:])
    except IndexError:
        args = parser.parse_args()  # use default values

    # accessing parsed args
    models = args.models
    training = args.training
    verbose = args.verbose
    plot = args.plot

    # run main code
    results = main(models, training, verbose)

    if plot:
        df = pd.DataFrame.from_dict(results)
        plot_loss(df)
        plot_accuracy(df)

    parser.exit(message="training complete\n")
