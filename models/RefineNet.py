import torch.nn as nn
import torch.nn.functional as F

class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class MultiResolutionFusion(nn.Module):

    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError(f"max_size not divisble by shape {i}")

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module(f"resolve{i}", nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                ))
            else:
                self.add_module(
                    f"resolve{i}",
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__(f"resolve{i}")(x)

        return output


class ChainedResidualPool(nn.Module):

    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module(f"block{i}", nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
            ))

    def forward(self, x):
        # x = self.relu(x)
        # path = x

        # for i in range(1, 4):
        #     path = self.__getattr__(f"block{i}")(path)
        #     x += path

        # return x
        x = self.relu(x)
        path = x

        path_1 = self.__getattr__("block1")(path)
        res_x = path + path_1

        path_2 = self.__getattr__("block2")(path)
        res_x = path + path_2

        path_3 = self.__getattr__("block3")(path)
        res_x = path + path_3
        # for i in range(1, 4):
        #     path = self.__getattr__(f"block{i}")(path)
        #     x += path

        return x


class ChainedResidualPoolImproved(nn.Module):

    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(f"block{i}", nn.Sequential(
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__(f"block{i}")(path)
            x += path

        return x


class BaseRefineNetBlock(nn.Module):

    def __init__(self, features,
                 residual_conv_unit,
                 multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(f"rcu{i}", nn.Sequential(
                residual_conv_unit(feats),
                residual_conv_unit(feats)
            ))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        for i, x in enumerate(xs):
            x = self.__getattr__(f"rcu{i}")(x)

        if self.mrf is not None:
            out = self.mrf(*xs)
        else:
            out = xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):

    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit,
                         MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

import torch.nn as nn
import torchvision.models as models

class BaseRefineNet4Cascade(nn.Module):

    def __init__(self, input_shape,
                 refinenet_block=RefineNetBlock,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super(BaseRefineNet4Cascade, self).__init__()
        self.model_name = "BaseRefineNet4Cascade"
        input_channel, input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError(f"{input_shape} not divisble by 32")

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBlock(
            2 * features, (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(
            features, (2 * features, input_size // 32), (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(
            features, (features, input_size // 16), (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(
            features, (features, input_size // 8), (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # self.softmax = nn.Softmax2d()

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        # out = self.softmax(out)
        return out


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):

    def __init__(self, input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(input_shape, RefineNetBlockImprovedPooling,
                         num_classes=num_classes, features=features,
                         resnet_factory=resnet_factory, pretrained=pretrained,
                         freeze_resnet=freeze_resnet)


class RefineNet4Cascade(BaseRefineNet4Cascade):

    def __init__(self, input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(input_shape, RefineNetBlock,
                         num_classes=num_classes, features=features,
                         resnet_factory=resnet_factory, pretrained=pretrained,
                         freeze_resnet=freeze_resnet)
        