import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data
def data_loader(mode):
    svhn_transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

    if mode == 'train':
        mnist_dset = dsets.MNIST(root='./mnist',
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))

        svhn_dset = dsets.SVHN(root='./svhn',
                   transform=svhn_transform)

        mnist_dset_loader = torch.utils.data.DataLoader(mnist_dset, batch_size=32,
                                                        shuffle=True, num_workers=2)
        svhn_dset_loader = torch.utils.data.DataLoader(svhn_dset, batch_size=32,
                                                       shuffle=True, num_workers=32)

        return mnist_dset_loader, svhn_dset_loader
    else:
        svhn_train_dset = dsets.SVHN(root='./svhn/extra',split='extra', download=True,
                               transform=svhn_transform)
        svhn_test_dset = dsets.SVHN(root='./svhn/test', split='test', download=True,
                                    transform=svhn_transform)

        svhn_train_dset_loader = torch.utils.data.DataLoader(svhn_train_dset, batch_size=64,
                                                       shuffle=True, num_workers=32)
        svhn_test_dset_loader = torch.utils.data.DataLoader(svhn_test_dset, batch_size=64,
                                                       shuffle=True, num_workers=32)

        return svhn_train_dset_loader, svhn_test_dset_loader