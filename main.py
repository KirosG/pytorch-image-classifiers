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
        batch_size=16,
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
                print(" Batch: %5d - Loss: %.3f" % (i+1, running_loss/100))
                print(" Accuracy: {:.2f}%".format(accuracy))

            running_loss = 0.0
    print()

    return scores


def main(epochs, training=True, verbose=True):
    net = LeNet()
    print(net, "\n")
    dataloader = load_data(training=training)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    results = defaultdict(list)
    for epoch in range(epochs):  # Our Epochs
        print("Epoch {}...".format(epoch+1))
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
        results['step'].append(step)
        results['loss_scores'].append(loss)
        results['acc_scores'].append(acc)
        results['model_n'].append([epoch] * len(step))

        if training:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Implementation of LeNet.",  # program title
    )
    parser.add_argument(
        '-epochs',  # argument name
        type=int,  # default data type
        help="The number of epoch to run",  # cli help description
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
    try:
        args = parser.parse_args(sys.argv[1:])
    except IndexError:
        args = parser.parse_args()  # use default values

    # accessing parsed args
    epochs = args.epochs
    training = args.training
    verbose = args.verbose

    results = main(epochs, training, verbose)

    parser.exit(message="Finished Training.\n")
