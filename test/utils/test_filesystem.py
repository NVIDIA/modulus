from pathlib import Path
from modulus.utils import filesystem


def test_glob(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    # use file:// protocol to ensure handling is correct
    (f,) = filesystem.glob(f"file://{tmp_path.as_posix()}/*.txt")
    assert f == f"file://{a.as_posix()}"


def test_glob_no_scheme(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    (f,) = filesystem.glob(f"{tmp_path.as_posix()}/*.txt")
    assert f == a.as_posix()
