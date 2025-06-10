import numpy as np
import torch


def reshape_w(mass):
    return mass.reshape((1, mass.shape[0], 1, 1))


def batch_norm(mass, gamma, beta, mean, std, eps=0.001):
    gamma = reshape_w(gamma)
    beta = reshape_w(beta)
    mean = reshape_w(mean)
    std = reshape_w(std)
    res = gamma * ((mass - mean) / np.sqrt(std + eps)) + beta
    return res


def unsq(mass):
    mass = torch.unsqueeze(mass, 1)
    mass = torch.unsqueeze(mass, 2)
    mass = torch.unsqueeze(mass, 3)
    return mass


def batchn_fusion(weight, gamma, beta, mean, std, eps=0.001):
    weight = weight.numpy().transpose(3, 0, 1, 2)
    # print(weight.shape, 'batchnorm weight')
    gamma = reshape_w(gamma.numpy())
    beta = reshape_w(beta.numpy())
    mean = reshape_w(mean.numpy())
    std = reshape_w(std.numpy())

    # print(weight.shape, gamma.shape)
    weightn = gamma * weight / np.sqrt(std + eps)
    biasn = ((gamma * (-mean)) / np.sqrt(std + eps)) + beta
    return torch.from_numpy(weightn.transpose(1, 2, 3, 0)), torch.from_numpy(biasn.reshape(weight.shape[1]))
