from torchvision.ops.misc import FrozenBatchNorm2d

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone,
    mobilenet_backbone,
    _validate_trainable_layers
)


def overwrite_eps(model, eps):
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}

resnet_fpn_backbones = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
]

mobilenet_backbones = ['MobileNetV2', 'mobilenet_v2', 'MobileNetV3', 'mobilenet_v3_large', 'mobilenet_v3_small']


def maskrcnn_model(backbone_name='resnet50', pretrained=False, progress=True,
                   num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    
    pretrained = pretrained and (backbone_name in model_urls.keys())

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[backbone_name],
                                              progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


if __name__ == '__main__':
    _ = maskrcnn_model()
