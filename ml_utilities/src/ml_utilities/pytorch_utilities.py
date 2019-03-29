import torch
import numpy as np


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model


def num_parameters(model):
    total_parameters = 0
    for p in model.parameters():
        total_parameters += np.prod(p.shape)
    return total_parameters
