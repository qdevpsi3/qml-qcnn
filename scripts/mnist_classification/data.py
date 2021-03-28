import torch
import torchvision


def get_loader(is_train,
               batch_size,
               folder='mnist_classification/datasets/',
               dataset='mnist',
               preprocess=((0.1307, ), (0.3081, ))):
    assert dataset in ['mnist', 'fashion'
                       ], "dataset can take values in ['mnist','fashion']"
    if dataset == 'mnist':
        dataset_class = torchvision.datasets.MNIST
    elif dataset == 'fashion':
        dataset_class = torchvision.datasets.FashionMNIST
    if preprocess is not None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*preprocess)
        ])
    else:
        transform = torchvision.transforms.ToTensor()
    dataset = dataset_class('{}/{}'.format(folder, dataset),
                            train=is_train,
                            download=True,
                            transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)
    return loader
