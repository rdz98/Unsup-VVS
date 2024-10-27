from augmentation import ColourDistortion
from dataset import *
from models import *


def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=None):
    CACHED_MEAN_STD = {'stl10': ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))}
    PATHS = {'stl10': './data/'}
    root = PATHS[dataset]

    # Data
    if dataset == 'stl10':
        img_size = 96
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])
    rp_transform = transforms.Normalize(*CACHED_MEAN_STD[dataset])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_clftrain = transform_test

    if dataset == 'stl10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.STL10)
        else:
            dset = torchvision.datasets.STL10
        if num_positive is None:
            trainset = STL10(biaugment=True, root=root, split='unlabeled', download=True,
                               transform=transform_train, rp_transform=rp_transform)
        else:
            raise NotImplementedError
        testset = STL10(biaugment=False, root=root, split='test', download=True,
                        transform=transform_test, rp_transform=rp_transform)
        clftrainset = dset(root=root, split='train', download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemSTL
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    return trainset, testset, clftrainset, num_classes, stem
