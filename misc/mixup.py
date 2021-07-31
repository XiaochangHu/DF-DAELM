import numpy as np
import torch

def mixup_data(args, x, y):
    batch_size = len(y)
    size = np.ones(len(x.shape)).astype(np.int)
    size[0] = batch_size
    lam = np.random.beta(args.alpha, args.alpha, batch_size).astype(np.float32)

    lam_x = torch.from_numpy(lam.reshape(size)).to(args.torch_device)
    lam_y = torch.from_numpy(lam.reshape(batch_size, 1)).to(args.torch_device)

    index = np.random.permutation(batch_size)
    x1, x2 = x, x[index]
    mixed_x = x1 * lam_x + x2 * (1 - lam_x)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam_y