import sys
import shutil
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop, RandomRotation
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

from models.lenet import LeNet
from visuals import plot_accuracy, plot_loss


USE_CUDA = torch.cuda.is_available()

sns.set_style('darkgrid')


def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)


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


def train(model, dataloader, criterion, optimizer, epoch):
    scores = []

    steps = []
    losses = []
    accuracies = []

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, unit='batch')  # iterable for progress bar

    for i, (inputs, labels) in enumerate(pbar):
        # wrap features as torch Variables
        if USE_CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels)  # optimization
        loss.backward()  # compute back propagation
        optimizer.step()  # update model parameters

        running_loss += loss.data[0]

        if i % 100 == 99:  # print every 100 mini-batches
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            accuracy = 100. * correct / total

            steps.append(i+1)
            losses.append(running_loss/100)
            accuracies.append(accuracy)

            pbar.set_description(
                "Epoch %2d, Accuracy %.2f%%, Progress" % (epoch, accuracy)
            )
            running_loss = 0.0  # zero the loss


    return steps, losses, accuracies


def main(epochs, training=True):
    results = defaultdict(list)

    net = LeNet()
    print(net)

    if USE_CUDA:  # GPU optimization
        net.cuda()
        net = torch.nn.DataParallel(
            net,
            device_ids=range(torch.cuda.device_count())
        )
        cudnn.benchmark = True

    dataloader = load_data(training=training)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    for i in range(epochs):

        steps, losses, acc = train(
            net,  # our model
            dataloader,  # the data provider
            criterion,  # the loss function
            optimizer,  # the optimization algorithm
            i+1,  # current epoch
        )

        # add observations to the dictionary
        results['step'] += steps
        results['loss_scores'] += losses
        results['acc_scores'] += acc
        results['epoch'] += [i+1] * len(steps)

        if training:
            save_checkpoint({
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            })

    return results


if __name__ == "__main__":
    if USE_CUDA:
        print("GPUs Detected. Accelerating.")

    parser = argparse.ArgumentParser(
        description="Image classifiers implemented with PyTorch",  # title
    )
    parser.add_argument(
        '-epochs',  # argument name
        type=int,  # default data type
        help="total epochs to run",  # cli help description
        required=False,  # does this need to be passed
        default=1,  # default value for the argument
    )
    parser.add_argument(
        '-training',
        type=bool,
        help='are you training?',
        required=False,
        default=True,
    )
    parser.add_argument(
        '-plot',
        type=bool,
        help='plot model results',
        required=False,
        default=False,
    )
    parser.add_argument(
        '-savefig',
        type=bool,
        help='save figures',
        required=False,
        default=False,
    )
    args = parser.parse_args()  # use default values

    # accessing parsed args
    epochs = args.epochs
    training = args.training
    plot = args.plot
    save_fig = args.savefig

    # run main code
    results = main(epochs, training)

    df = pd.DataFrame.from_dict(results)
    loss_ = plot_loss(df, save_fig)
    acc_ = plot_accuracy(df, save_fig)

    if plot:
        print("Plotting results...")
        plt.show()
