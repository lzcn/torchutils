import torch
from torch import nn


class ConditionalBatchNorm2d(nn.Module):
    r"""Applies Conditional Batch Normalization over a 4D input.

    .. math::

        y=\frac{x-\mathbb{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}} * \gamma_i+\beta_i



    where :math:`i` is the class of the input :math:`x`.

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        num_classes (int): :math:`S` the number of conditions
        eps (float, optional): a value added to the denominator for numerical stability.
            Default to 1e-5.
        momentum (float, optional): the value used for the ``running_mean`` and
            ``running_var`` computation. Can be set to ``None`` for cumulative
            moving average (i.e. simple average). Default to 0.1.


    Examples:

        >>> m = layers.ConditionalBatchNorm2d(100, 10)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> c = torchutils.one_hot(torch.randint(10, size=(20, 1)), 10)
        >>> output = m(input, c)

    """

    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.normalize = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
        self.weight = nn.Parameter(torch.ones(num_classes, num_features))
        self.bias = nn.Parameter(torch.zeros(num_classes, num_features))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute the conditional batch normalization.
        Args:
            x (torch.Tensor): data of shape :math:`(N, C, H, W)`
            c (torch.Tensor): conditions of shape :math:`(N, S)`

        Returns:
            [torch.Tensor]: output of shape :math:`(N, C, H, W)`
        """
        h = self.normalize(x)
        w = torch.matmul(c, self.weight).view(-1, self.num_features, 1, 1)
        b = torch.matmul(c, self.bias).view(-1, self.num_features, 1, 1)
        y = w * h + b
        return y


class LinerLN(nn.Module):
    """Linear layer with LayerNorm, ReLU, and Dropout.

    Args:
        in_features: size of each input sample
        out_features:  size of each output sample
        bias: If set to `False`, the layer will not learn an additive bias. Default: `True`
        dropout: probability of an element to be zeroed. Default: 0.0
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.ln = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: output
        """
        x = self.fc(x)
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x
