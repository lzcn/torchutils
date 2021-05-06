from torchutils import io

LIST_DATA = [
    ["0", "1", "2", "3", "4"],
    ["0", "1", "2", "3", "4"],
    ["0", "1", "2", "3", "4"],
]
INT_DATA = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 4],
]
DICT_DATA = {"0": 0, "1": 1, "2": 2, "3": 3}


def test_json(tmp_path):
    d = tmp_path / "json"
    d.mkdir()
    fn = d / "data.json"
    io.save_json(fn, LIST_DATA)
    data = io.load_json(fn)
    assert data == LIST_DATA
    io.save_json(fn, DICT_DATA)
    data = io.load_json(fn)
    assert data == DICT_DATA


def test_csv(tmp_path):
    d = tmp_path / "json"
    d.mkdir()
    fn = d / "data.json"
    io.save_csv(fn, INT_DATA)
    data = io.load_csv(fn)
    assert data == LIST_DATA


def test_csv_convert(tmp_path):
    d = tmp_path / "json"
    d.mkdir()
    fn = d / "data.json"
    io.save_csv(fn, LIST_DATA)
    data = io.load_csv(fn, converter=int)
    assert data == INT_DATA


def test_csv_skip(tmp_path):
    d = tmp_path / "json"
    d.mkdir()
    fn = d / "data.json"
    num_skip = 1
    io.save_csv(fn, LIST_DATA)
    data = io.load_csv(fn, num_skip=num_skip, converter=int)
    assert data == INT_DATA[num_skip:]
