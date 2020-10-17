import numpy as np
import pytest
from torchutils.meter import BundleMeter, Meter


def _generate_data(size, win_size=0):
    xs = np.random.randn(size)
    ws = np.random.randn(size)
    ys = xs * ws
    ys_cum = ys.cumsum()
    ws_cum = ws.cumsum()
    if win_size > 0:
        ys_cum[win_size:] = ys_cum[win_size:] - ys_cum[:-win_size]
        ws_cum[win_size:] = ws_cum[win_size:] - ws_cum[:-win_size]
    vs = ys_cum / ws_cum
    return xs, ws, vs


def test_meter():
    for win_size in range(0, 100, 20):
        meter = Meter(win_size)
        for size in range(win_size + 1, win_size + 100):
            meter.reset()
            xs, ws, vs = _generate_data(size, win_size)
            for x, w, v in zip(xs, ws, vs):
                meter.update(x, weight=w)
                assert meter.avg == pytest.approx(v)
                assert meter.val == pytest.approx(x)


def test_bundle_meter():
    num_data = 10
    for win_size in range(0, 100, 20):
        meter = BundleMeter(win_size)
        for size in range(win_size + 1, win_size + 100):
            meter.reset()
            # generate 10 sequences
            data = [_generate_data(size, win_size) for _ in range(num_data)]
            for i in range(size):
                data_dict = dict()
                results = dict()
                for n in range(num_data):
                    xs, ws, zs = data[n]
                    meter.update({str(n): xs[i]}, weight=ws[i])
                    results[str(n)] = zs[i]
                    data_dict[str(n)] = xs[i]
                assert list(results.keys()) == list(meter.avg.keys())
                assert list(results.values()) == pytest.approx(list(meter.avg.values()))
                assert list(data_dict.values()) == pytest.approx(list(meter.val.values()))
