import torch
import torchvision
import torchvision.transforms as transforms


# MNIST data loader (10 classes)
def get_mnist_data_loaders(batch_size=32, root_path="data/", download=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    X_train = torchvision.datasets.MNIST(root=root_path, train=True, download=download, transform=transform)
    X_train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    X_test = torchvision.datasets.MNIST(root=root_path, train=False, download=download, transform=transform)
    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    return(X_train_loader, X_test_loader, X_train.classes)

# Cifar10 data loader (10 classes)
def get_cifar10_data_loaders(batch_size=32, root_path="data/", download=False):
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    X_train = torchvision.datasets.CIFAR10(root=root_path, train=True, download=download, transform=transform)
    X_train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    X_test = torchvision.datasets.CIFAR10(root=root_path, train=False, download=download, transform=transform)
    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    return(X_train_loader, X_test_loader, X_train.classes)
