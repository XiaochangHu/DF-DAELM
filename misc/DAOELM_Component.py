import torch
from cupy._core.dlpack import toDlpack
from cupy._core.dlpack import fromDlpack
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import cupy as cp
import time
import sys

class DataAugmented_OELM(object):
    def __init__(self, hidden_layer_params, gpu_device=None, class_number=10):
        if gpu_device is not None:
            # Set the gpu for cupy. if gpu_device == None, the cupy use the same id of gpu as that of torch.
            cp.cuda.Device(gpu_device).use()
        self.params_shape = hidden_layer_params[len(hidden_layer_params) - 1].shape
        self.class_number = class_number
        self.xtx = []
        self.w = cp.zeros(self.params_shape, dtype=cp.float32).T
        self.I_ = cp.identity(self.params_shape[1],dtype=cp.float32)


    # traditional L2-OS-ELM
    def L2_OSELM_Init(self, inputs, targets, l2_regular_weight=0.42):

        self.C=l2_regular_weight
        self.I = self.C * self.I_

        inputs = fromDlpack(to_dlpack(inputs))
        targets = fromDlpack(to_dlpack(targets))
        targets = self.OnehotVectorize(targets=targets)
        self.xtx = cp.dot(inputs.T, inputs) + self.I
        w = cp.dot(cp.linalg.pinv(self.xtx), inputs.T)
        self.w = cp.dot(w, targets)
        w = []
        targets = []

    def L2_OSELM_Iter(self, inputs, targets):
        inputs = fromDlpack(to_dlpack(inputs))
        targets = fromDlpack(to_dlpack(targets))
        targets = self.OnehotVectorize(targets=targets)
        self.xtx = self.xtx + cp.dot(inputs.T, inputs)
        h = cp.dot(cp.linalg.inv(self.xtx), cp.dot(inputs.T, targets - cp.dot(inputs, self.w)))
        self.w += h
        h = []
        targets = []

    #  CR-OELM
    def CR_OELM_Init(self, inputs, inputs_, targets, l2_regular_weight=0.42, cr_regular_weight=None):
        self.C=l2_regular_weight
        self.C1=cr_regular_weight

        if(cr_regular_weight == None):
            print("The weight of consistency regularization are not set correctly")
            sys.exit(1)

        self.I = self.C * self.I_

        inputs = fromDlpack(to_dlpack(inputs))
        inputs_ = fromDlpack(to_dlpack(inputs_))
        targets = fromDlpack(to_dlpack(targets))

        targets = self.OnehotVectorize(targets=targets)
        self.xtx = cp.dot(inputs.T, inputs) + self.C1*cp.dot((inputs-inputs_).T, (inputs-inputs_)) + self.I  # [ n_features * n_features ]
        w = cp.dot(cp.linalg.pinv(self.xtx), inputs.T)
        self.w = cp.dot(w, targets)
        pppp = cp.dot(w, targets)
        w = []
        targets = []

    def CR_OELM_Iter(self, inputs, inputs_, targets):
        inputs = fromDlpack(to_dlpack(inputs))
        inputs_ = fromDlpack(to_dlpack(inputs_))
        targets = fromDlpack(to_dlpack(targets))

        targets = self.OnehotVectorize(targets=targets)
        self.xtx = self.xtx + cp.dot(inputs.T, inputs) + self.C1*cp.dot((inputs-inputs_).T, (inputs-inputs_))
        h = cp.dot(cp.linalg.inv(self.xtx),(cp.dot(inputs.T,targets)-
                                  cp.dot((cp.dot(inputs.T,(1+self.C1)*inputs-2*self.C1*inputs_)
                                  +self.C1*cp.dot(inputs_.T,inputs_)),self.w)))
        self.w += h
        h = []
        targets = []

    #  SLI-OELM
    def SLI_OELM_Init(self, inputs, targets_a, targets_b, l2_regular_weight=0.42, lam=None):
        # targets_a/b is a  n*classnumber dim vector, e.g., 1,2,3,4, which will be converted to onehot labels by the OnehotVectorize function.
        # inputs is a torch tensor, which reprensets the hidden output matrix.
        # lam is the interpolated weight vector
        if(lam == None):
            print("The parameters of the Beta distribution of the stochastic linear difference are not set correctly")
            sys.exit(1)
        self.C=l2_regular_weight
        self.I = self.C * self.I_

        inputs = fromDlpack(to_dlpack(inputs))
        targets_a = fromDlpack(to_dlpack(targets_a))
        targets_b = fromDlpack(to_dlpack(targets_b))
        lam = fromDlpack(to_dlpack(lam))

        targets_a = self.OnehotVectorize(targets=targets_a)
        targets_b = self.OnehotVectorize(targets=targets_b)

        self.xtx = cp.dot(inputs.T, inputs) + self.I  # [ n_features * n_features ]
        w = cp.dot(cp.linalg.pinv(self.xtx), inputs.T)
        self.w = cp.dot(w, (lam*targets_a+(1-lam)*targets_b))
        w = []
        targets_a = []
        targets_b = []

    def SLI_OELM_Iter(self, inputs, targets_a, targets_b, lam):
        # targets_a/b is n*class_number  dim, e.g., 1,2,3,4, which will be converted to onehot labels by the OnehotVectorize function.
        # inputs is a torch tensor, which reprensets the hidden output matrix.
        # lam is the interpolated weight vector

        inputs = fromDlpack(to_dlpack(inputs))
        targets_a = fromDlpack(to_dlpack(targets_a))
        targets_b = fromDlpack(to_dlpack(targets_b))
        lam = fromDlpack(to_dlpack(lam))

        targets_a = self.OnehotVectorize(targets=targets_a)
        targets_b = self.OnehotVectorize(targets=targets_b)
        self.xtx = self.xtx + cp.dot(inputs.T, inputs)
        h = cp.dot(cp.linalg.inv(self.xtx), cp.dot(inputs.T, lam*targets_a+(1-lam)*targets_b - cp.dot(inputs, self.w)))
        self.w += h
        h = []
        targets_a = []
        targets_b = []


    # Convert labels to one-hot encoding form
    def OnehotVectorize(self, targets):
        oneHotTarget=cp.zeros((targets.shape[0], self.class_number), dtype=cp.float32)
        for i in range(targets.shape[0]):
            if targets[i] != -1:
                oneHotTarget[i][targets[i]]=1
        return oneHotTarget

    def output(self, output_weight_params=None):
        if(output_weight_params == None):
            print("The output weight parameter setting is wrong！！！")
            sys.exit(1)
        par = list(output_weight_params)
        par[len(par) - 1].data = from_dlpack(toDlpack(self.w.T))

if __name__ == "__main__":
    W = torch.zeros(500,2500).cuda()
    input = torch.rand(500,2500).cuda()
    input_ = torch.rand(500, 2500).cuda()
    target = torch.tensor([0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0,5,1,2,3,4,5,3,5,3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           0, 5, 1, 2, 3, 4, 5, 3, 5, 3,
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]).cuda()
    eg = DataAugmented_OELM(hidden_layer_params=W, class_number=10)
    init = time.time()
    for i in  range(1):
        end = time.time()

        if i == 0:
            eg.CR_OELM_Init(input, input_, target, l2_regular_weight = 0.42, cr_regular_weight=1)
        else:
            eg.CR_OELM_Iter(input, input_, target)
    print(end - init)
    init = end
    eg.output(output_weight_params=W)
    print(W)
