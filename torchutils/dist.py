import numpy as np
import torch
from scipy import linalg
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.inception import inception_v3
from tqdm import tqdm
from torch import nn


@torch.no_grad()
def _inception_output(dataset, model, batch_size=128, num_workers=4, resize=True, progress=False, device="cpu"):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    dataloader = tqdm(dataloader) if progress else dataloader
    outputs = []
    for batch in dataloader:
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, list):
            x = batch[0]
        else:
            raise TypeError("The return of dataset can not be recognized")
        x = x.to(device)
        if resize:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        outputs.append(model(x))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def inception_score(dataset, batch_size=32, num_workers=4, splits=1, resize=True, progress=False, device="cpu"):
    r"""Compute the inception score (IS)

    .. math::

        \log(\text{IS}) = \mathbb{E}_{x\sim p_{data}} D_{KL}\big(p(y|x)|| p(y)\big)

    Examples:

        .. code-block:: python

            dataset = datasets.ImageNet(
                    "data/imagenet",
                    split="val",
                    transform=transforms.Compose(
                        [
                            transforms.Resize(299),
                            transforms.CenterCrop(299),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ]
                    ),
                )
            mean, std = inception_score(dataset, resize=False)
            print("{:.3f}({:.3f})".format(mean, std)) # 63.653(8.227)

    Args:
        dataset (Dataset): dataset.
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): number of workers. Defaults to 4.
        splits (int, optional): number of splits. Defaults to 1.
        resize (bool, optional): whether to resize image. Defaults to False.
        progress (bool, optional): show progress bar. Defaults to False.
        device (str, optional): [description]. Defaults to "cpu".

    Returns:
        tuple: mean IS and std IS

    """
    assert splits > 0
    model = inception_v3(pretrained=True, init_weights=False)
    model.to(device)
    model.eval()
    output = _inception_output(dataset, model, batch_size, num_workers, resize, progress, device)
    probs = F.softmax(output, dim=-1).split(len(dataset) // splits)

    scores = []
    for pr in probs:
        prior = torch.mean(pr, dim=0, keepdim=True).expand_as(pr)
        kl_div = F.kl_div(prior.log(), pr, reduction="batchmean")
        scores.append(torch.exp(kl_div))
    scores = torch.stack(scores, dim=0)
    mean = torch.mean(scores).item()
    std = torch.std(scores).item() if splits > 1 else 0.0
    return mean, std


@torch.no_grad()
def fid_score(dataset1, dataset2, batch_size=32, num_workers=4, resize=True, progress=False, device="cpu"):
    r"""Compute the FID score.

    .. math::

        \text{FID}=|\mu_{1}-\mu_{2}|^{2}+\text{tr}\big(\Sigma_{1}+\Sigma_{2}-2(\Sigma_{1}\Sigma_{2})^{1/2}\big)

    Args:
        dataset1 ([Dataset, Tensor]): datasets to compare fid score
        dataset2 ([Dataset, Tensor]): datasets to compare fid score
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): number of workers. Defaults to 4.
        resize (bool, optional): whether to resize image. Defaults to False.
        progress (bool, optional): show progress bar. Defaults to False.
        device (str, optional): [description]. Defaults to "cpu".

    Returns:
        float: FID score
    """
    model = inception_v3(pretrained=True, init_weights=False)
    model.fc = nn.Identity()
    x1 = _inception_output(dataset1, model, batch_size, num_workers, resize, progress, device)
    x2 = _inception_output(dataset2, model, batch_size, num_workers, resize, progress, device)
    x1 = x1.cpu().numpy()
    x2 = x2.cpu().numpy()
    m1 = x1.mean(axis=0)
    m2 = x2.mean(axis=0)
    c1 = np.cov(x1.T)
    c2 = np.cov(x2.T)
    sqrtm, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    score = np.sum((m1 - m2) ** 2) + np.trace(c1 + c2 - 2.0 * sqrtm)
    return score


def make_1D_gauss(n, mean=0, std=1.0, norm=True):
    """return a 1D histogram for a gaussian distribution

    Parameters
    ----------
    n : int
        number of bins in the histogram
    mean : float
        mean value of the gaussian distribution
    std : float
        standard deviaton of the gaussian distribution

    Returns
    -------
    h : ndarray (n,)
        1D histogram for a gaussian distribution
    """
    x = np.arange(n, dtype=np.float64)
    h = np.exp(-0.5 * ((x - mean) ** 2) / (std ** 2))
    if norm:
        return h / h.sum()
    else:
        return h
