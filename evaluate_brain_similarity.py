from models import ResNet18, StemSTL
import torch
import torch.backends.cudnn as cudnn
import functools
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import load_benchmark
import argparse
import pickle
import os

LAYERS = ['layer1.0.relu', 'layer1.1.relu', 'layer2.0.relu', 'layer2.1.relu',
          'layer3.0.relu', 'layer3.1.relu', 'layer4.0.relu', 'layer4.1.relu']
BENCHMARKS = ['FreemanZiemba2013public.V1-pls', 'FreemanZiemba2013public.V2-pls',
              'MajajHong2015public.V4-pls', 'MajajHong2015public.IT-pls']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Brain Similarity.')
    parser.add_argument("--load-from", type=str, default='ckpt', help='File to load from')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet18(stem=StemSTL).to(device)
    if device == 'cuda':
        repr_dim = net.representation_dim
        net = torch.nn.DataParallel(net)
        net.representation_dim = repr_dim
        cudnn.benchmark = True

    checkpoint = torch.load(f'checkpoint/{args.load_from}.pth')
    net.load_state_dict(checkpoint['net'])

    preprocessing = functools.partial(load_preprocess_images, image_size=96)
    wrapper = PytorchWrapper(identifier=args.load_from, model=net.module, preprocessing=preprocessing)
    wrapper.image_size = 96

    for i in BENCHMARKS:
        benchmark = load_benchmark(i)
        score_dict = {}

        for j in LAYERS:
            model = ModelCommitment(identifier=f'{args.load_from}_{j}', activations_model=wrapper, layers=[j])
            score_dict[j] = benchmark(model)

        dir_path = f'brain_score/{i}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(f'{dir_path}/{args.load_from}.pkl', 'wb') as f:
            pickle.dump(score_dict, f)
