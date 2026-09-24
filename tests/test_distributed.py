import torchutils as tu


def test_rank_zero_only_runs_without_distributed():
    calls = []

    @tu.rank_zero_only
    def fn(x):
        calls.append(x)
        return x * 2

    assert fn(21) == 42
    assert calls == [21]
