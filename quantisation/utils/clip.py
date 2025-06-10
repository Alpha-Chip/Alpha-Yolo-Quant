def clip(mass, k):
    mass[mass <= -(2 ** (k - 1) - 1)] = -(2 ** (k - 1) - 1)
    mass[mass >= (2 ** (k - 1) - 1)] = (2 ** (k - 1) - 1)
    return mass