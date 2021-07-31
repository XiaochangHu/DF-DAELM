import torch
import torch.utils.data.dataloader
from misc.mixup import mixup_data

import os

# stochastic linear interpolation online extreme learning machine (SLI-OELM)
def train_SLI_OELM(args, model, elm_model, train_loader, epoch = 0, batch_init_id=0):

    for batch_idx, (data, target) in enumerate(train_loader):
        if epoch == 0:
            if batch_idx == batch_init_id:
                data, target = data.to(args.torch_device), target.to(args.torch_device)
                # stochastic linear interpolation

                data, labels_a, labels_b, lam = mixup_data(args, data, target)
                hiddenOut = model.forwardToHidden(data)
                elm_model.SLI_OELM_Init(inputs=hiddenOut, targets_a=labels_a, targets_b=labels_b, l2_regular_weight=args.c, lam=lam)
                break
        else:
            data, target = data.to(args.torch_device), target.to(args.torch_device)
            # stochastic linear interpolation
            data, labels_a, labels_b, lam = mixup_data(args, data, target)
            hiddenOut = model.forwardToHidden(data)
            elm_model.SLI_OELM_Iter(inputs=hiddenOut, targets_a=labels_a, targets_b=labels_b, lam=lam)

# consistency regularization online extreme learning machine (SLI-OELM)
def train_CR_OELM(args, model, elm_model, train_loader,  epoch = 0, batch_init_id=0):

    for batch_idx, (data, data1, target) in enumerate(train_loader):
        if epoch == 0:
            if batch_idx == batch_init_id:
                data, data1, target = data.to(args.torch_device), data1.to(args.torch_device), target.to(
                    args.torch_device)
                hiddenOut = model.forwardToHidden(data)
                hiddenOut1 = model.forwardToHidden(data1)

                elm_model.CR_OELM_Init(inputs=hiddenOut, inputs_= hiddenOut1, targets=target, l2_regular_weight=args.c, cr_regular_weight=args.c1)
        else:
            data, data1, target = data.to(args.torch_device), data1.to(args.torch_device), target.to(args.torch_device)
            hiddenOut = model.forwardToHidden(data)
            hiddenOut1 = model.forwardToHidden(data1)
            elm_model.CR_OELM_Iter(inputs=hiddenOut, inputs_= hiddenOut1, targets=target)


# l2 regularization online sequence extreme learning machine (L2-OS-ELM)
def train_L2_OELM(args, model, elm_model, train_loader,  epoch = 0, batch_init_id=0):

    for batch_idx, (data, target) in enumerate(train_loader):
        if epoch == 0:
            if batch_idx == batch_init_id:
                data, target = data.to(args.torch_device),  target.to(args.torch_device)
                hiddenOut = model.forwardToHidden(data)
                elm_model.L2_OSELM_Init(inputs=hiddenOut, targets=target, l2_regular_weight=args.c)
                break
        else:
            data, target = data.to(args.torch_device), target.to(args.torch_device)
            hiddenOut = model.forwardToHidden(data)
            elm_model.L2_OSELM_Iter(inputs=hiddenOut, targets=target)

# test
def test(args, model, elm_model, test_loader):
    elm_model.output(list(model.parameters()))
    model.eval()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(args.torch_device), target.to(args.torch_device)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum().numpy()
    acc = float(correct) * 100.0 / float(len(test_loader.dataset))
    return acc

def save_checkpoint(state, dirpath,dataset_scale_ratio,lable_noise_ratio):
    filename = 'checkpoint.ratio:{:.2f}_noise:{:.2f}_best.ckpt'.format(dataset_scale_ratio,lable_noise_ratio)
    checkpoint_path = os.path.join(dirpath, filename)
    torch.save(state, checkpoint_path)