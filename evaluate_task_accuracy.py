import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn

from configs import get_datasets
from evaluate import encode_train_set, train_clf, test
from models import *
from classifier_head import ClassifierHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(20240630)
    parser = argparse.ArgumentParser(description='Evaluate Task Accuracy.')
    parser.add_argument("--num-workers", type=int, default=2, help='Number of threads for data loaders')
    parser.add_argument("--load-from", type=str, nargs='+', default=['ckpt'], help='File to load from')
    args = parser.parse_args()

    # Load checkpoint.
    print('==> Loading settings from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = {}
    for i in args.load_from:
        resume_from = os.path.join('./checkpoint', i + '.pth')
        checkpoint = torch.load(resume_from)
        args.dataset = checkpoint['args']['dataset']
        args.arch = checkpoint['args']['arch']

        # Data
        print('==> Preparing data..')
        _, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

        testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
        clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True)

        # Model
        print('==> Building model..')
        ##############################################################
        # Encoder
        ##############################################################
        if args.arch == 'resnet18':
            net = ResNet18(stem=stem)
        else:
            raise ValueError("Bad architecture specification")
        net = net.to(device)

        if device == 'cuda':
            repr_dim = net.representation_dim
            net = torch.nn.DataParallel(net)
            net.representation_dim = repr_dim
            cudnn.benchmark = True

        print('==> Loading encoder from checkpoint..')
        net.load_state_dict(checkpoint['net'])

        predictor = ClassifierHead(net.representation_dim * 2).to(device)
        predictor.load_state_dict(checkpoint['predictor'])

        X, y = encode_train_set(clftrainloader, device, net)
        clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
        acc, acc2 = test(testloader, device, net, clf, predictor)
        result[i] = (acc, acc2)
    for i, (acc, acc2) in result.items():
        print(f'Test accuracy of {i}, {acc}% / {acc2}%')
