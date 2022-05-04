import os
import random

import torch.nn as nn
from torchutils.io import ModelSaver

model = nn.Linear(1, 1)


def test_min_saver(tmp_path):
    n_saved = 5
    num_epochs = 10
    scores = sorted([random.random() for _ in range(num_epochs)])
    epochs = list(range(num_epochs))
    saver = ModelSaver(
        tmp_path,
        filename_prefix="net",
        score_name="score",
        n_saved=n_saved,
        mode="min",
        save_latest=True,
        save_best=True,
    )
    for score, epoch in zip(scores, epochs):
        saver.save(model, score, epoch)
    for n in range(num_epochs):
        score = scores[n]
        epoch = epochs[n]
        d = tmp_path / f"net_score_{score:.4f}_epoch_{epoch}.pt"
        if n < n_saved:
            assert os.path.isfile(d)
        else:
            assert not os.path.isfile(d)
    assert saver.last_checkpoint == os.path.join(tmp_path, f"net_score_{scores[0]:.4f}_epoch_{epochs[0]}.pt")
    assert os.path.isfile(tmp_path / "net_latest.pt")
    assert os.path.isfile(tmp_path / "net_best.pt")


def test_max_saver(tmp_path):
    n_saved = 5
    num_epochs = 10
    scores = sorted([random.random() for _ in range(num_epochs)], reverse=True)
    epochs = list(range(num_epochs))
    saver = ModelSaver(
        tmp_path,
        filename_prefix="net",
        score_name="score",
        n_saved=n_saved,
        mode="max",
        save_latest=True,
        save_best=True,
    )
    for score, epoch in zip(scores, epochs):
        saver.save(model, score, epoch)
    for n in range(num_epochs):
        score = scores[n]
        epoch = epochs[n]
        d = tmp_path / f"net_score_{score:.4f}_epoch_{epoch}.pt"
        if n < n_saved:
            assert os.path.isfile(d)
        else:
            assert not os.path.isfile(d)
    assert saver.last_checkpoint == os.path.join(tmp_path, f"net_score_{scores[0]:.4f}_epoch_{epochs[0]}.pt")
    assert os.path.isfile(tmp_path / "net_latest.pt")
    assert os.path.isfile(tmp_path / "net_best.pt")
