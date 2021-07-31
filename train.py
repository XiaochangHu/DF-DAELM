import argparse
import torch.utils.data.dataloader
from misc import  DAOELM_Component as DAELM
from misc import  DAOELM as tainer
from misc import architectures
from misc.dataset import modified_mnist
import torchvision
import torchvision.transforms as transforms
import sys
import random
import numpy as np
import time

parser = argparse.ArgumentParser(description='PyTorch and Cupy DAO-ELM MNIST Example')

parser.add_argument('--init-batch-size', type=int, default=400,help='input batch size for training')
parser.add_argument('--batch-size', type=int, default=400,help='input batch size for training')
parser.add_argument('--test-batch-size', type=int, default=1000,help='input batch size for testing (default: 10000)')
parser.add_argument('--epochs', type=int, default=50,help='iterative training epochs')

parser.add_argument('--dataset-path', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='mnist', help='mnist')
parser.add_argument('--class-num', type=int, default=10, help='the class number of the dataset')
parser.add_argument('--label-noise-level', type=float, default=0.8, help='Proportion of samples containing noisy labels')
parser.add_argument('--scale-ratio-level', type=float, default=0.1666, help='hidden size')

parser.add_argument('--train_subdir', type=str, default='training.pt')
parser.add_argument('--eval_subdir', type=str, default='test.pt')

parser.add_argument('--ELM', type=str, default='SLI-OELM', help='SLI-OELM, CR-OELM, L2-OSELM')
parser.add_argument('--architecture', type=str, default='Net_liner', help='the arch Single hidden layer neural network for ELM')
parser.add_argument('--hidden_size', type=int, default=1200, help='hidden size')
parser.add_argument('--activation', type=str, default='leaky_relu', help='non-linear activation (default: leaky_relu. you can try relu, leaky_relu, tanhshrink, softsign, selu')
parser.add_argument('--c', type=float, default=0.01, help='elm parameter weight for SLI-OELM/ L2-OELM/ CR-OELM')
parser.add_argument('--c1', type=float, default=0.42, help='elm parameter weight for CR-OELM')
parser.add_argument('--alpha', type=float, default=0.6, help='elm parameter weight for SLI-OELM')

parser.add_argument('--torch-gpu', type=int, default=0, help='gpu device for torch')
parser.add_argument('--cupy-gpu', type=int, default=0, help='gpu device for copy')

parser.add_argument('--seed', type=int, default=4444,help='random seed')
args = parser.parse_args()

torch_device = torch.device("cuda:{}".format(args.torch_gpu))
args.torch_device = torch_device

# set seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# a demo of setting dataset
# note that you can try other data aumentation methods via adjusting the transformation design.
if args.dataset == 'mnist':
    train_transformation = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    if args.ELM == 'CR-OELM':
        train_transformation1 = transforms.Compose(
            [transforms.RandomRotation(degrees=60),
             transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])
    else:
        train_transformation1 = None

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    dataset_train = modified_mnist(args=args, transform=train_transformation, transform1 = train_transformation1)
    dataset_test = torchvision.datasets.MNIST(root=args.dataset_path, train=False, transform=eval_transformation)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
else:
    print('if you want to use other dataset, such as cifar10, you need to prepare its dataloder ')
    sys.exit(0);

# set model
model_factory = architectures.__dict__[args.architecture]
model_params = dict(num_classes=args.class_num, hidden_size=args.hidden_size)
model = model_factory(**model_params)
model = model.to(args.torch_device)

# set elm
elm_model = DAELM.DataAugmented_OELM(hidden_layer_params=list(model.parameters()), gpu_device=args.cupy_gpu, class_number=10)

# for SLI-OELM
if args.ELM == 'SLI-OELM':
    init = time.time()
    for epoch in range(args.epochs):
        tainer.train_SLI_OELM(args, model, elm_model, train_loader, epoch, batch_init_id = 1)
        end = time.time()
        acc = tainer.test(args, model, elm_model, test_loader)

        print('SLI-OELM -- [{}\{}], test accurcy: {:.4f}%, time: {:.4f}s'.format(epoch, args.epochs, acc, (end-init)))
        init = end

# for CR-OELM
if args.ELM == 'CR-OELM':
    init = time.time()
    for epoch in range(args.epochs):
        tainer.train_CR_OELM(args, model, elm_model, train_loader, epoch, batch_init_id = 1)
        end = time.time()
        acc = tainer.test(args, model, elm_model, test_loader)
        print('CR-OELM -- [{}\{}], test accurcy: {:.4f}%, time: {:.4f}s'.format(epoch, args.epochs, acc,  (end-init)))
        init = end

# L2-OS-ELM for comparation
if args.ELM == 'L2-OSELM':
    init = time.time()
    for epoch in range(args.epochs):
        tainer.train_L2_OELM(args, model, elm_model, train_loader, epoch, batch_init_id = 1)
        end = time.time()
        acc = tainer.test(args, model, elm_model, test_loader)
        print('L2-OS-ELM -- [{}\{}], test accurcy: {:.4f}%, time: {:.4f}s'.format(epoch, args.epochs, acc,  (end-init)))
        init = end