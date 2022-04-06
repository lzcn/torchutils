import os

import torch.nn as nn

from torchutils.io import ModelSaver

model = nn.Linear(1, 1)


def test_saver_epoch(tmp_path):
    n_saved = 5
    num_epochs = 10
    saver = ModelSaver(tmp_path, filename_prefix="net", score_name="score", n_saved=n_saved)
    for epoch in range(num_epochs):
        score = 1.0 * epoch
        saver.save(model, 1.0 * epoch, epoch)
        d = tmp_path / f"net_score_{score:.4f}_epoch_{epoch}.pt"
        assert os.path.isfile(d)
    for epoch in range(num_epochs - n_saved):
        score = 1.0 * epoch
        d = tmp_path / f"net_score_{score:.4f}_epoch_{epoch}.pt"
        assert not os.path.isfile(d)
    assert saver.last_checkpoint == os.path.join(tmp_path, f"net_score_{num_epochs-1:.4f}_epoch_{num_epochs-1}.pt")
