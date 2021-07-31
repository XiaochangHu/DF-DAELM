
import torch
import torch.utils.data as data
import random
import numpy as np
from PIL import Image
import os
import os.path
import sys
import torchvision

class modified_mnist(data.Dataset):
    def __init__(self, args, transform=None, transform1=None):
        self.classnum = args.class_num
        self.transform = transform
        self.transform1 = transform1
        mnist = torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True)
        self.data_original, self.targets_original = mnist.data, mnist.targets
        self.data, self.targets ,self.targets_indicator= self.data_original, self.targets_original, self.targets_original
        self.noise_ratio(scale_ratio_level=args.scale_ratio_level, label_noise_level=args.label_noise_level)

    def __getitem__(self, index):
        img, target ,indicator= self.data[index], self.targets[index], self.targets_indicator[index]
        Img = Image.fromarray(img.numpy(), mode='L')
        if self.transform1 is not None:
            Img1 = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            Img = self.transform(Img)
        if self.transform1 is not None:
            Img1 = self.transform1(Img1)
            return Img, Img1, target
        else:
            return Img, target

    def __len__(self):
        return len(self.data)

    def dataset_analy(self, ratio=1):
        length_data = len(self.data_original)
        length_tar = len(self.targets_original)
        if length_data != length_tar:
            raise RuntimeError('Dataset is bad')
        else:
            list0 = list(range(length_tar))
            random.shuffle(list0)
            end = int(length_tar*ratio)
            list0 = sorted(list0[:end])
        num = np.zeros(self.classnum)
        for i in list0:
            num[self.targets_original[i]] +=1
        return list0

    def noise_ratio(self, scale_ratio_level=1, label_noise_level=1):
        if (scale_ratio_level>0 and scale_ratio_level<=1):
            list = self.dataset_analy(ratio=scale_ratio_level)
            self.data = self.data_original[list]
            self.targets = self.targets_original[list]
        else:
            raise RuntimeError('the ratio of datasets is error, note that 0<ratio<=1')

        K, self.targets, self.targets_indicator  = addRandomNoiseToTrainingSet(self.targets, label_noise_level, self.classnum)
        print("There are {} samples (total {} samples) used to trained model which has {}% nosiy labels".format(len(self.targets), len(self.targets_original), label_noise_level*100))
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    targets = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    images.append(path)
                    targets.append(class_to_idx[target])
    return np.array(images), np.array(targets)

def find_classes(dir):
    """
    Finds the class folders in a dataset.

    Args:
        dir (string): Root directory path.

    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

    Ensures:
        No class is a subdirectory of another.
    """
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def addRandomNoiseToTrainingSet(lables, noise_level, classnumber):
    label_data_set = lables
    totalNum = label_data_set.shape[0]
    label_data_indi = label_data_set
    k = int(noise_level * totalNum)
    if noise_level<0 or noise_level>1:
        raise RuntimeError('the noise level of label is error, note that 0=<noise level<=1')
    if noise_level>0 and noise_level< 1:
        corruptedIdxList = randomSelectKFromN(k, totalNum)
        for cIdx in corruptedIdxList:
            correctLabel = label_data_set[cIdx]
            wrongLabel = convertCorrectLabelToCorruptedLabel(correctLabel, classnumber)
            label_data_set[cIdx] = wrongLabel
            label_data_indi[cIdx] = -1

        return k, label_data_set, label_data_indi
    else:
        return k, lables, label_data_indi

def randomSelectKFromN(K, N):
    resultList = []
    seqList = list(range(N))
    random.shuffle(seqList)
    list0 = sorted(seqList[:K])
    return list0

def convertCorrectLabelToCorruptedLabel(correctLabel, classnumber):
    correct_value = correctLabel
    target_value = int(np.random.rand(1)[0] * classnumber)
    if target_value == correct_value:
        target_value = int((target_value + 1))
        if target_value>=classnumber:
            target_value = target_value - classnumber
    return target_value


