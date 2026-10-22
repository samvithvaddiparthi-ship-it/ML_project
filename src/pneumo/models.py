"""Model definitions.

Both classifiers end in   conv feature map -> Global Average Pool -> one Linear
layer, which is exactly the structure Class Activation Mapping (CAM, Zhou et
al. 2016) needs:  M_c(x, y) = sum_k  w_k^c * f_k(x, y).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMClassifier(nn.Module):
    """Interface: `features(x)` -> (B, C, h, w);  `fc` is Linear(C, 1)."""

    fc: nn.Linear

    def features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        return self.fc(F.adaptive_avg_pool2d(f, 1).flatten(1)).squeeze(1)

    @torch.no_grad()
    def cam(self, x: torch.Tensor):
        """Return (logits (B,), CAM maps (B, h, w)) for the pneumonia class."""
        f = self.features(x)
        logits = self.fc(F.adaptive_avg_pool2d(f, 1).flatten(1)).squeeze(1)
        w = self.fc.weight[0]                       # (C,)
        cams = torch.einsum("c,bchw->bhw", w, f)
        return logits, cams


class PaperCNN(CAMClassifier):
    """Re-implementation of the Stanford CS229 model: 10 conv layers, all 3x3,
    zero padding, ReLU, then GAP + a single FC layer.  We add BatchNorm and
    three 2x2 max-pools (the report does not specify down-sampling); at a
    128x128 input the final map is 16x16, which keeps the CAM reasonably sharp.
    """

    def __init__(self, in_ch: int = 1, widths=(32, 32, 64, 64, 128, 128, 256, 256, 256, 256),
                 pool_after=(1, 3, 5)):
        super().__init__()
        layers, c = [], in_ch
        for i, w in enumerate(widths):
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            if i in pool_after:
                layers.append(nn.MaxPool2d(2))
            c = w
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(c, 1)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        f = self.features(x)
        return self.fc(self.drop(F.adaptive_avg_pool2d(f, 1).flatten(1))).squeeze(1)


class DenseNetCAM(CAMClassifier):
    """DenseNet-121 (the CheXNet backbone), ImageNet-pretrained, with a
    GAP + Linear(1024, 1) head so CAM can be computed directly."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import densenet121, DenseNet121_Weights
        net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = net.features
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, 1)

    def features(self, x):
        return F.relu(self.backbone(x))

    def forward(self, x):
        f = self.features(x)
        return self.fc(self.drop(F.adaptive_avg_pool2d(f, 1).flatten(1))).squeeze(1)


def build_classifier(name: str, pretrained: bool = True) -> CAMClassifier:
    if name == "PaperCNN":
        return PaperCNN()
    if name == "DenseNet121":
        return DenseNetCAM(pretrained=pretrained)
    raise ValueError(name)


def build_detector(pretrained: bool = True, size: int = 512):
    """torchvision Faster R-CNN, ResNet-50-FPN, COCO-pretrained, 2 classes."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    kw = dict(min_size=size, max_size=size, box_detections_per_img=10)
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, **kw)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, **kw)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
