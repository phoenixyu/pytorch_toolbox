import torchvision.models as models

from .FCN import *
from .RefineNet import *
from .SegNet import *
from .DeepUNet import *

def get_model(name, n_classes):

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = fcn8s(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == 'refinenet':
        model = BaseRefineNet4Cascade((3,128),
                num_classes=n_classes,
                features=256,
                resnet_factory=models.resnet101,
                pretrained=True,
                freeze_resnet=True)
    elif name == 'segnet':
        model = segnet(n_classes=n_classes,
                      is_unpooling=True)
    elif name == 'deepunet':
        model = deepunet(n_channels=3, n_classes=n_classes)
    else:
        raise 'Model {} not available'.format(name)

    return model