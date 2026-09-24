import torchutils as tu


def test_default_scan_skips_hidden_and_subdirs(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / ".DS_Store").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").touch()

    assert len(tu.scan_files(tmp_path)) == 2
    assert tu.scan_files(tmp_path, "jpg") == [str(tmp_path / "a.jpg")]


def test_suffix_forms_and_recursive(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").touch()

    assert len(tu.scan_files(tmp_path, (".jpg", ".png"))) == 2
    assert sorted(tu.scan_files(tmp_path, "jpg", recursive=True, relpath=True)) == [
        "a.jpg",
        "sub/c.jpg",
    ]
    assert tu.scan_files(tmp_path, "txt") == []
