import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # torch functions


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # run __init__ of parent class
        # Convolutional Layers
        # 1 image, 6 output channels, 5x5 convolution
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))

        # Fully Connected Layers

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        """
        forward must be overwritten in torch model class
        """
        # Convolutional Layers
        ## add pooling layers
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 256)  # flatten to pass to fully connected layers

        # fully connected layers
        ## and dropout layers
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        """
        return the number of flat features from a pytorch variable
        """
        return int(np.prod(x.size()[1:]))


def test():
    net = LeNet()
    print(net)
    dims = (1,1,28,28)
    print("Dimensions: " + " x ".join([str(d) for d in dims][1:]))
    x = torch.randn(dims)
    y = net(x)
    print("Classes:", y.size()[1])


if __name__ == "__main__":
    test()
