from .alexnet import alexnet
from .squeezenet import squeezenet1_0, squeezenet1_1
from .densenet import densenet121, densenet161, densenet169, densenet201
from .vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from .resnet import *
from .need_train_model import mnist_model, cifar_model, svhn_model
from .cifar_vgg import cifar_vgg16, cifar_vgg19, cifar_vgg19_bn
from .resnet_CBAM import *
from .cifar_resnet_CBMA import *
from .cifar_resnet import *

