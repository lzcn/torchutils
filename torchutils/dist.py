import numpy as np
import torch
from scipy import linalg
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.inception import inception_v3
from tqdm.auto import tqdm


@torch.no_grad()
def _inception_output(
    dataset, model, transform_input=False, batch_size=128, num_workers=4, resize=True, progress=False, device=None
):
    if device is None:
        # auto-select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    dataloader = tqdm(dataloader, desc="Forward Inception V3") if progress else dataloader
    outputs = []
    for batch in dataloader:
        if isinstance(batch, torch.Tensor):
            x = batch
        elif isinstance(batch, list):
            x = batch[0]
        else:
            raise TypeError("The return of dataset can not be recognized")
        x = x.to(device)
        if transform_input:
            x = normalize(x / 2.0 + 0.5)
        if resize:
            # if align corners IS=10.254 else IS=10.511 on CIFAR10
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=True)
        outputs.append(model(x))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def inception_score(
    dataset, transform_input=False, batch_size=32, num_workers=4, splits=1, resize=True, progress=False, device=None
):
    r"""Compute the inception score (IS)

    .. math::

        \log(\text{IS}) = \mathbb{E}_{x\sim p_{data}} D_{KL}\big(p(y|x)|| p(y)\big)

    Examples:

        .. code-block:: python

            from torchvision import datasets
            from torchutils.dist import inception_score

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
            mean, std = inception_score(dataset, resize=False, progress=True)
            print("{:.3f}({:.3f})".format(mean, std))

    Args:
        dataset (Dataset): dataset.
        transform_input (bool, optional): whether to transform the input from range
            (-1, 1) to desired range. Defaults to False.
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): number of workers. Defaults to 4.
        splits (int, optional): number of splits. Defaults to 1.
        resize (bool, optional): whether to resize image. Defaults to True.
        progress (bool, optional): show progress bar. Defaults to False.
        device (str, optional): device to compute. Defaults to None.

    Returns:
        tuple: mean IS and std IS

    """
    assert splits > 0
    model = inception_v3(pretrained=True, transform_input=False)
    model.to(device)
    model.eval()
    output = _inception_output(dataset, model, transform_input, batch_size, num_workers, resize, progress, device)
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


def _inception_mean_and_cov(dataset, model, batch_size, num_workers, resize, progress, device):
    x = _inception_output(dataset, model, batch_size, num_workers, resize, progress, device)
    x = x.cpu().numpy()
    mean = x.mean(axis=0)
    cov = np.cov(x.T)
    return mean, cov


@torch.no_grad()
def fid_score(
    dataset_1=None,
    dataset_2=None,
    mu_1=None,
    mu_2=None,
    cov_1=None,
    cov_2=None,
    transform_input=False,
    batch_size=32,
    num_workers=4,
    resize=True,
    progress=False,
    device=None,
    return_mean_and_cov=False,
):
    r"""Compute the FID score.

    .. math::

        \text{FID}=|\mu_{1}-\mu_{2}|^{2}+\text{tr}\big(\Sigma_{1}+\Sigma_{2}-2(\Sigma_{1}\Sigma_{2})^{1/2}\big)

    Args:
        dataset_1 ([Dataset, Tensor]): datasets to compare fid score
        dataset_2 ([Dataset, Tensor]): datasets to compare fid score
        batch_size (int, optional): batch size. Defaults to 32.
        num_workers (int, optional): number of workers. Defaults to 4.
        resize (bool, optional): whether to resize image. Defaults to True.
        progress (bool, optional): show progress bar. Defaults to False.
        device (str, optional): device to compute. Defaults to None.

    Returns:
        float: FID score
    """
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    if mu_1 is None or cov_1 is None:
        mu_1, cov_1 = _inception_mean_and_cov(
            dataset_1, model, transform_input, batch_size, num_workers, resize, progress, device
        )
    if mu_2 is None or cov_2 is None:
        mu_2, cov_2 = _inception_mean_and_cov(
            dataset_2, model, transform_input, batch_size, num_workers, resize, progress, device
        )
    sqrtm, _ = linalg.sqrtm(cov_1.dot(cov_2), disp=False)
    score = np.sum((mu_1 - mu_2) ** 2) + np.trace(cov_1 + cov_2 - 2.0 * sqrtm)
    if return_mean_and_cov:
        return score, {"mu_1": mu_1, "mu_2": mu_2, "cov_1": cov_1, "cov_2": cov_2}
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
    h = np.exp(-0.5 * ((x - mean) ** 2) / (std**2))
    if norm:
        return h / h.sum()
    else:
        return h
