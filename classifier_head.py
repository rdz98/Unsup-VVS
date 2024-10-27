import torch
from torch import nn


class ClassifierHead(nn.Module):

    def __init__(self, latent_dim):
        super(ClassifierHead, self).__init__()
        self.projection_dim = 8
        self.w1 = nn.Linear(latent_dim, 512)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(512, self.projection_dim)

    def forward(self, h1, h2):
        h = torch.cat([h1, h2], dim=-1)
        return self.w2(self.relu(self.w1(h)))
