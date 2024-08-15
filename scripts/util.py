import os
import torchgeometry
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.autograd import Variable
from sklearn.preprocessing import scale
import torch.fft as fft
import math
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import random
import torch.backends.cudnn as cudnn
from global_variable import base_dir
import sys
sys.path.append(base_dir)

from models import (cifar_vgg16, cifar_vgg19, cifar_resnet18, cifar_resnet152)
from MAD import ResNet18 as mad_cifar_resnet18
from MAD import ResNet34 as mad_cifar_resnet34
from models import (
    mnist_model,
    cifar_model,
    svhn_model,
    alexnet,
    squeezenet1_0,
    squeezenet1_1,
    densenet121,
    densenet169,
    densenet201,
    densenet161,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


list_nets = [
    'mnist_model',
    'cifar_model',
    'svhn_model',
    'alexnet',
    'squeezenet1_0',
    'squeezenet1_1',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
     'Cifar_VGG']


STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.220, 'bim-b': 0.230},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.175, 'bim-a': 0.070, 'bim-b': 0.105}
}

max_intensity = 255

vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    def rfft(x, d):
        t = torch.fft.fft(x, dim = (-d))
        r = torch.stack((t.real, t.imag), -1)
        return r
    def irfft(x, d):
        t = torch.fft.ifft(torch.complex(x[:,:,0], x[:,:,1]), dim = (-d))
        return t.real


# Set random seed
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(args):
    transformer = __get_transformer(args)
    dataset = __get_dataset_name(args)
    trn_loader, dev_loader, tst_loader = __get_loader(args, dataset, transformer)

    return trn_loader, dev_loader, tst_loader


def get_subsample_loader(args, loader, indices):
    subsample_loader = torch.utils.data.DataLoader(
        Subset(loader.dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True
    )
    return subsample_loader


def __get_loader(args, data_name, transformer):
    root = os.path.join(f'{base_dir}/data', data_name)
    data_path = os.path.join(root, args.dataset.lower())
    dataset = getattr(torchvision.datasets, data_name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # set transforms

    trn_transform = transformer
    tst_transform = transformer

    if data_name == 'SVHN':
        trainset = dataset(
            root=data_path, download=True, split='train', transform=trn_transform
        )
        tstset = dataset(
            root=data_path, download=True, split='test', transform=tst_transform
        )
    elif data_name == 'ImageNet':
        traindir = os.path.join(data_path, 'ILSVRC2012_img_train')
        valdir = os.path.join(data_path, 'ILSVRC2012_img_val')

        trainset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize
            ])
        )
        tstset = datasets.ImageFolder(
            valdir,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize
            ])
        )

    else:
        trainset = dataset(
            root=data_path, download=True, train=True, transform=trn_transform
        )
        tstset = dataset(
            root=data_path, download=True, train=False, transform=tst_transform
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    devloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    tstloader = torch.utils.data.DataLoader(
        tstset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    
    print(len(trainloader.dataset), len(devloader.dataset), len(tstloader.dataset))

    return trainloader, devloader, tstloader

def __get_transformer(args):
    if args.dataset == 'mnist':
        return transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    else : 
        return transforms.Compose([transforms.ToTensor()])

def __get_dataset_name(args):
    if args.dataset.lower() == "mnist":
        d_name = "MNIST"
    elif args.dataset.lower() == "fmnist":
        d_name = "FashionMNIST"
    elif args.dataset.lower() == "cifar":
        d_name = "CIFAR10"
    elif args.dataset.lower() == "cifar100":
        d_name = "CIFAR100"
    elif args.dataset.lower() == "svhn":
        d_name = "SVHN"
    elif args.dataset.lower() == "imagenet":
        d_name = "ImageNet"
    return d_name


def get_imagenet_classifier(classifier_name, pretrained):
    """Load converted model"""

    if classifier_name == 'alexnet':
        classifier = alexnet(pretrained=pretrained)
    elif classifier_name == 'squeezenet1_0':
        classifier = squeezenet1_0(pretrained=pretrained)
    elif classifier_name == 'squeezenet1_1':
        classifier = squeezenet1_1(pretrained=pretrained)
    elif classifier_name == 'densenet121':
        classifier = densenet121(pretrained=pretrained)
    elif classifier_name == 'densenet169':
        classifier = densenet169(pretrained=pretrained)
    elif classifier_name == 'densenet201':
        classifier = densenet201(pretrained=pretrained)
    elif classifier_name == 'densenet161':
        classifier = densenet161(pretrained=pretrained)
    elif classifier_name == 'vgg11':
        classifier = vgg11(pretrained=pretrained)
    elif classifier_name == 'vgg13':
        classifier = vgg13(pretrained=pretrained)
    elif classifier_name == 'vgg16':
        classifier = vgg16(pretrained=pretrained)
    elif classifier_name == 'vgg19':
        classifier = vgg19(pretrained=pretrained)
    elif classifier_name == 'vgg11_bn':
        classifier = vgg11_bn(pretrained=pretrained)
    elif classifier_name == 'vgg13_bn':
        classifier = vgg13_bn(pretrained=pretrained)
    elif classifier_name == 'vgg16_bn':
        classifier = vgg16_bn(pretrained=pretrained)
    elif classifier_name == 'vgg19_bn':
        classifier = vgg19_bn(pretrained=pretrained)
    elif classifier_name == 'resnet18':
        classifier = resnet18(pretrained=pretrained)
    elif classifier_name == 'resnet34':
        classifier = resnet34(pretrained=pretrained)
    elif classifier_name == 'resnet50':
        classifier = resnet50(pretrained=pretrained)
    elif classifier_name == 'resnet101':
        classifier = resnet101(pretrained=pretrained)
    elif classifier_name == 'resnet152':
        classifier = resnet152(pretrained=pretrained)
    elif classifier_name == 'mnist_model':
        classifier = mnist_model
    elif classifier_name == 'cifar_model':
        classifier = cifar_model
    elif classifier_name == 'svhn_model':
        classifier = svhn_model
    else:
        print('Wrong model name:', classifier_name, '!')
        exit()
    classifier = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier, )

    return classifier, classifier_name


def get_cifar_classifier(classifier_name, num_classes, pretrained):
    if classifier_name == 'vgg16':
        classifier = cifar_vgg16(num_classes=num_classes)
    elif classifier_name == 'vgg19':
        classifier = cifar_vgg19(num_classes=num_classes)
    elif classifier_name == 'resnet18':
        classifier = cifar_resnet18(num_classes=num_classes)
    elif classifier_name == 'resnet152':
        classifier = cifar_resnet152(num_classes=num_classes)
    elif classifier_name == 'mad_resnet18':
        classifier = mad_cifar_resnet18(num_c=10)
    elif classifier_name == 'mad_resnet34':
        classifier = mad_cifar_resnet34(num_c=10)
    else:
        print('Wrong model name:', classifier_name, '!')
        exit()
    return classifier, classifier_name


class Normalize(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()

        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class AdvDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, attack, one_three, net_name, eps):
        
        self.dataset = dataset
        self.attack = attack
        self.data_dict = torch.load(f'{base_dir}/adv_data/{dataset}/{net_name}/{one_three}/{attack}/eps_{eps}.pkl')

    def __len__(self):
        return len(self.data_dict['X'])

    def __getitem__(self, idx):
        x, y = self.data_dict['X'][idx], self.data_dict['Y'][idx]
        return x, y


def get_adv_loader(args, dataset, attack, one_three, net_name, eps):
    loader = torch.utils.data.DataLoader(
        AdvDataset(dataset, attack, one_three, net_name, eps),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
        )
    return loader


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def compute_roc(probs, labels, plot=False):
    """
    TODO
    :param probs:
    :param labels:
    :param plot:
    :return:
    """
    # probs = np.concatenate((probs_neg, probs_pos))
    # labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='red',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


def evaluate(model, loader, criterion, device, return_pred=False):
    model.eval()
    with torch.no_grad():
        true = []
        pred = []
        test_loss = 0
        for i, data in enumerate(loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            # outputs = nn.Softmax(dim=1)(outputs)
            loss = criterion(outputs, labels)

            pred.append(outputs.argmax(dim=1))
            true.append(labels)
            test_loss += len(inputs)*loss
        true = torch.cat(true, dim=0)
        pred = torch.cat(pred, dim=0)
        correct_predictions = pred.eq(true).sum()
        accuracy = correct_predictions / len(loader.dataset) * 100
        if return_pred:
            return test_loss.cpu().numpy()/len(loader.dataset), accuracy.cpu().numpy(), pred, true
        else:
            return test_loss.cpu().numpy()/len(loader.dataset), accuracy.cpu().numpy()



def local_histogram_equalization(tensor_img, n):
    # equalized_images = []
    # tensor_img = RGB2gray(tensor_img)
    # numpy_array = tensor_img.cpu().detach().numpy()
    # print(numpy_array.shape)
    # for i in range(numpy_array.shape[0]):
    #     single_image = np.uint8(np.array(numpy_array[i]))
    #     # equalized_img = localEq(single_image, n)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(n, n))
    #     equalized_img = clahe.apply(single_image)
    #     equalized_tensor = torch.from_numpy(equalized_img).unsqueeze(0).float() / 255.0
    #     equalized_images.append(equalized_tensor.unsqueeze(0))
    # equalized_images_tensor = torch.cat(equalized_images, dim=0)
    # return equalized_images_tensor
    # 转换图像范围从 [0, 1] 到 [0, 255]
    batch_images = (tensor_img * 255).to(torch.uint8).cpu().numpy()

    # 对每张图像的每个通道分别进行局部直方图均衡化
    equalized_images = []
    channels = []
    for i in range(batch_images.shape[0]):
        image = batch_images[i]  # 获取一张图像
        for j in range(image.shape[0]):
            channel = image[j]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            channels.append(clahe.apply(channel))

        # 合并通道
        equalized_image = cv2.merge((channels[0], channels[1], channels[2]))
        channels = []
        # 将处理后的图像转换回 PyTorch 张量
        equalized_tensor = torch.from_numpy(equalized_image).unsqueeze(0).float() / 255.0
        equalized_images.append(equalized_tensor)
    equalized_images_tensor = torch.cat(equalized_images, dim=0)
    equalized_images_tensor = equalized_images_tensor.permute(0, 3, 1, 2)
    return equalized_images_tensor











