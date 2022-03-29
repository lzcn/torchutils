import torch
from torch import nn

from .loss import contrastive_loss


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


class VisualSemanticEmbedding(nn.Module):
    """The visual-semantic embedding layer with :meth:`torchutils.loss.contrastive_loss`

    Cosine similarity is used.

    Args:
        i_dim (int): dimension for visual data
        t_dim (int): dimension for semantic data or the size of vocabulary.
        c_dim (int): dimension for embedding space
        margin (float, optional): margin for loss. Defaults to ``0.2``.
        bow (bool, optional): whether the input is bag-of-word. Defaults to ``False``.
    """

    def __init__(self, i_dim, t_dim, c_dim, margin=0.2, bow=False):
        super().__init__()
        self.i_dim = i_dim
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.margin = margin
        self.bow = bow
        self.Wi = nn.Linear(i_dim, c_dim, bias=False)
        self.Wt = nn.Linear(t_dim, c_dim, bias=False)

    def forward(self, i_data: torch.Tensor, t_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute the visual-semantic contrastive loss.

        Args:
            i_data (torch.Tensor): image data with shape :math:`(B, N, D)`
                or :math:`(B, D)`.
            t_data (torch.Tensor): text data with shape :math:`(B, N, D)`
                or :math:`(B, D)`.
            mask (torch.Tensor, optional): mask for data of shape :math:`(B, N)`
                or :math:`(B,)`. Defaults to None.

        Returns:
            torch.Tensor: margin loss between two embedding sets.
        """
        assert len(i_data) == len(t_data) == len(mask)
        *shape, _ = i_data.shape
        if self.bow:
            # normalize by number of words
            t_norm = t_data.sum(dim=-1, keepdim=True)
            t_data = t_data / t_norm.clamp_min(1e-10)
            # mask for whether it has any word
            t_mask = t_norm > 0.0
            if mask is None:
                mask = t_mask
            else:
                mask = mask * t_mask
        else:
            mask = None if mask is None else mask.view(-1, 1)
        i_feat = self.Wi(i_data).view(-1, self.c_dim)
        t_feat = self.Wt(t_data).view(-1, self.c_dim)
        loss = contrastive_loss(i_feat, t_feat, self.margin, mask=mask, reduction="none")
        return loss.view(*shape)
