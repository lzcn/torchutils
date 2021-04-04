import pytest
import torch
import torchutils


def get_support_backbones():
    from torchutils.backbones import _BACKBONES

    return _BACKBONES.keys()


@pytest.mark.parametrize("name", get_support_backbones())
def test_backbone_loading(name):
    torchutils.backbone(name, pretrained=True)


@pytest.mark.parametrize("name", ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
def test_resnet_affine(name):
    resnet, resnet_dim = torchutils.backbone(name, pretrained=True)
    affine, affine_dim = torchutils.backbone(name + "_affine", pretrained=True)
    assert resnet_dim == affine_dim
    resnet_state = resnet.state_dict()
    affine_state = affine.state_dict()
    for key, resnet_param in resnet_state.items():
        if key in affine_state:
            affine_param = affine_state[key]
            if isinstance(resnet_param, torch.LongTensor):
                if resnet_param.numel() > 1:
                    assert all(affine_param == resnet_param)
                else:
                    assert affine_param == resnet_param
            else:
                diff = (affine_param - resnet_param).mean()
                assert diff == pytest.approx(0.0)
