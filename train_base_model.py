import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchlars import LARS
from tqdm import tqdm

from configs import get_datasets
from projection_head import ProjectionHead
from classifier_head import ClassifierHead
from evaluate import save_checkpoint, encode_train_set, train_clf, test
from models import *
from scheduler import CosineAnnealingWithLinearRampLR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    critic.train()
    predictor.train()
    train_loss = 0
    t = tqdm(enumerate(trainloader), desc='Loss: **** ', total=len(trainloader), bar_format='{desc}{bar}{r_bar}')
    for batch_idx, ((x1, x2, x3, x4), (_, y2), _) in t:
        x1, x2 = x1.to(device), x2.to(device)
        representation1, representation2 = net(x1), net(x2)
        raw_scores, pseudotargets = critic(representation1, representation2)
        loss = criterion(raw_scores, pseudotargets)

        x3, x4, y2 = x3.to(device), x4.to(device), y2.to(device)
        representation3, representation4 = net(x3), net(x4)
        pred_y = predictor(representation3, representation4)
        loss += args.alpha * criterion(pred_y, y2)

        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()

        train_loss += loss.item()
        t.set_description('Loss: %.3f ' % (train_loss / (batch_idx + 1)))


if __name__ == '__main__':
    set_seed(20240630)
    parser = argparse.ArgumentParser(description='Unsupervised Training.')
    parser.add_argument('--base-lr', default=1.5, type=float, help='base learning rate, rescaled by batch_size/256')
    parser.add_argument("--momentum", default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--resume', '-r', type=str, default='', help='resume from checkpoint with this filename')
    parser.add_argument('--dataset', '-d', type=str, default='stl10', help='dataset', choices=['stl10'])
    parser.add_argument('--temperature', type=float, default=0.5, help='InfoNCE temperature')
    parser.add_argument("--batch-size", type=int, default=512, help='Training batch size')
    parser.add_argument("--num-epochs", type=int, default=500, help='Number of training epochs')
    parser.add_argument("--cosine-anneal", action='store_true', help="Use cosine annealing on the learning rate")
    parser.add_argument("--arch", type=str, default='resnet18', help='Encoder architecture', choices=['resnet18'])
    parser.add_argument("--num-workers", type=int, default=12, help='Number of threads for data loaders')
    parser.add_argument("--test-freq", type=int, default=50,
                        help='Frequency to fit a linear clf with L-BFGS for testing'
                             'Not appropriate for large datasets. Set 0 to avoid '
                             'classifier only training here.')
    parser.add_argument("--filename", type=str, default='ckpt', help='Output file name')
    parser.add_argument("--alpha", default=0.01, type=float, help='alpha')
    args = parser.parse_args()
    args.lr = args.base_lr * (args.batch_size / 256)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    clf = None

    print('==> Preparing data..')
    trainset, testset, clftrainset, num_classes, stem = get_datasets(args.dataset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)
    clftrainloader = torch.utils.data.DataLoader(clftrainset, batch_size=1000, shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

    print('==> Building model..')
    ##############################################################
    # Base Model
    ##############################################################
    if args.arch == 'resnet18':
        net = ResNet18(stem=stem)
    else:
        raise ValueError("Bad architecture specification")
    net = net.to(device)

    ##############################################################
    # Projection Head for Contrastive Learning
    ##############################################################
    critic = ProjectionHead(net.representation_dim, temperature=args.temperature).to(device)

    ##############################################################
    # Classifier Head for Relative Position Learning
    ##############################################################
    predictor = ClassifierHead(net.representation_dim * 2).to(device)

    if device == 'cuda':
        repr_dim = net.representation_dim
        net = torch.nn.DataParallel(net)
        net.representation_dim = repr_dim
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        resume_from = os.path.join('./checkpoint', args.resume)
        checkpoint = torch.load(resume_from)
        net.load_state_dict(checkpoint['net'])
        critic.load_state_dict(checkpoint['critic'])
        predictor.load_state_dict(checkpoint['predictor'])
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    base_optimizer = optim.SGD(list(net.parameters()) + list(critic.parameters()) + list(predictor.parameters()),
                               lr=args.lr, weight_decay=1e-6, momentum=args.momentum)
    if args.cosine_anneal:
        scheduler = CosineAnnealingWithLinearRampLR(base_optimizer, args.num_epochs)
    encoder_optimizer = LARS(base_optimizer, trust_coef=1e-3)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch)
        if epoch % args.test_freq == (args.test_freq - 1):
            X, y = encode_train_set(clftrainloader, device, net)
            clf = train_clf(X, y, net.representation_dim, num_classes, device, reg_weight=1e-5)
            acc, acc2 = test(testloader, device, net, clf, predictor)
            save_checkpoint(net, clf, critic, predictor, epoch, acc, acc2, args, os.path.basename(__file__))
        if args.cosine_anneal:
            scheduler.step()
