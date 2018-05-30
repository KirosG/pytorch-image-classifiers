import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomCrop, RandomRotation, Resize, Grayscale

from models.lenet import LeNet
from visuals import plot_accuracy, plot_loss


USE_CUDA = torch.cuda.is_available()

sns.set_style('darkgrid')


def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)


def mnist_data(training=True):
    transform_ = transforms.Compose([
        Grayscale(),
        Resize(28),
        RandomRotation(45),
        RandomCrop(28),
        transforms.ToTensor(),
    ])
    data = torchvision.datasets.MNIST(
        root='./data/',
        train=training,
        download=True,
        transform=transform_,
    )
    # data = torchvision.datasets.CIFAR10(
    #     root='./data/',
    #     train=training,
    #     download=True,
    #     transform=transform_,
    # )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=128,
        shuffle=True,
        num_workers=8,
    )

    return loader


def train(model, dataloader, criterion, optimizer, epoch=0):
    steps = []
    losses = []
    accuracies = []

    running_loss = 0.0
    total = 0
    correct = 0

    desc = "Epoch %2d, Accuracy %.0f%%, Progress"

    pbar = tqdm(dataloader, unit='batch', desc=desc % (epoch+1, np.NaN))

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

        running_loss += loss.item()

        if i % 100 == 99:  # print every 100 mini-batches
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            accuracy = 100. * correct / total

            steps.append(i+1)
            losses.append(running_loss/100)
            accuracies.append(accuracy)

            pbar.set_description(
                desc % (epoch+1, accuracy)
            )
            running_loss = 0.0  # zero the loss

    return steps, losses, accuracies


def main(epochs, training=True, use_cuda=USE_CUDA):
    results = defaultdict(list)

    model = LeNet()
    print(model)

    if use_cuda:  # GPU acceleration
        model.cuda()
        model = torch.nn.DataParallel(
            model,
            device_ids=range(torch.cuda.device_count())
        )
        cudnn.benchmark = True

    dataloader = mnist_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        steps, losses, acc = train(
            model,  # the model
            dataloader,  # the data provider
            criterion,  # the loss function
            optimizer,  # the optimization algorithm
            epoch,  # current epoch
        )

        # add observations to the dictionary
        results['step'] += steps
        results['loss_scores'] += losses
        results['acc_scores'] += acc
        results['epoch'] += [epoch+1] * len(steps)

        if save:
            save_checkpoint({
                'state_dict': model.state_dict(),
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
        '--save_model',
        action='store_true',
        help='save model checkpoints',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='plot model results',
    )
    parser.add_argument(
        '--save_fig',
        action='store_true',
        help='save figures',
    )

    args = parser.parse_args()  # use default values

    # accessing parsed args
    epochs = args.epochs
    save = args.save_model
    plot = args.plot
    save_fig = args.save_fig

    # run main code
    results = main(epochs, save)

    df = pd.DataFrame.from_dict(results)
    loss_ = plot_loss(df, save_fig)
    acc_ = plot_accuracy(df, save_fig)

    if plot:
        print("Plotting results...")
        plt.show()
