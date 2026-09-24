import os

import pytest
import torch
import torch.nn as nn

import torchutils as tu


def test_eviction_best_and_latest(tmp_path):
    saver = tu.ModelSaver(tmp_path, n_saved=2, save_latest=True)
    net = nn.Linear(2, 2)
    for score, epoch in [(0.5, 1), (0.9, 2), (0.7, 3)]:
        saver.save(net, score=score, epoch=epoch)

    scored = [f for f in os.listdir(tmp_path) if f not in ("best.pt", "latest.pt")]
    assert len(scored) == 2
    entries = [f for _, f in saver.history if f]
    assert len(entries) == len(set(entries))
    assert all(os.path.exists(f) for f in entries)
    assert saver.best_checkpoint == os.path.join(tmp_path, "best.pt")
    assert torch.load(saver.best_checkpoint, weights_only=True) is not None


def test_duplicate_score_does_not_self_delete(tmp_path):
    saver = tu.ModelSaver(tmp_path, n_saved=2)
    net = nn.Linear(2, 2)
    saver.save(net, score=0.9, epoch=2)
    saver.save(net, score=0.9, epoch=2)

    path = tmp_path / "0.9000_epoch_2.pt"
    assert path.exists()
    assert [f for _, f in saver.history if f] == [str(path)]


def test_load_pretrained_unwraps_full_checkpoint(tmp_path):
    net = nn.Linear(2, 2)
    path = tmp_path / "ck.pt"
    torch.save({"state_dict": net.state_dict(), "epoch": 9, "score": 0.9}, path)

    other = nn.Linear(2, 2)
    tu.load_pretrained(other, path)
    assert torch.equal(other.weight, net.weight)


def test_load_pretrained_shape_mismatch_and_strict():
    net = nn.Linear(2, 2)
    mismatched = nn.Linear(3, 2)
    tu.load_pretrained(mismatched, net.state_dict())
    assert not torch.equal(mismatched.weight, net.weight)
    with pytest.raises(RuntimeError):
        tu.load_pretrained(mismatched, net.state_dict(), strict=True)


def test_invalid_saver_args(tmp_path):
    with pytest.raises(ValueError):
        tu.ModelSaver(tmp_path, n_saved=0)
    with pytest.raises(ValueError):
        tu.ModelSaver(tmp_path, mode="avg")
