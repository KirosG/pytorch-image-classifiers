import torch
import torchvision
import torchvision.transforms as transforms


def test_load():
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    data = torchvision.datasets.MNIST(
        root='./data/',
        train=False,
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


def main():
    dataloader = test_load()
    print(dataloader)

    pass


if __name__ == "__main__":
    print("Testing...")
    main()