import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from models.lenet import LeNet


def train_load():
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    data = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    return loader


def main(model):
    dataloader = train_load()
    print(dataloader)

    net = model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    print("Training...")
    main(LeNet)
