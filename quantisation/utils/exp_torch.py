import torch


def exponent_quant(x, lookup, device):
    ret = x
    k = torch.asarray(list(lookup.keys()))
    v = torch.asarray(list(lookup.values()))
    sidx = k.argsort()
    k = k[sidx]
    v = v[sidx]
    k = k.to(device)
    v = v.to(device)
    idx = torch.searchsorted(k, ret.ravel()).reshape(ret.shape)
    idx[idx == len(k)] = 0
    mask = k[idx] == ret
    ret = torch.where(mask, v[idx], 0)

    return ret





# def silu(x):
#     return (x * (1 / (1 + (np.e**(-x)))))


# 539894

#  275885834

# 8040.78576915085    0.0009464820872245293
# print(silu(31))
# print(31 / 14.131720713981359, 1/(np.e**(-31) + 1))
