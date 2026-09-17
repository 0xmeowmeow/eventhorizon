import tomllib

import pytest

from luminet import config


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_config_is_written_once_and_then_left_alone():
    cfg, problem = config.load_config()
    assert problem is None and cfg["start"] == "random" and cfg["unfocused_fps"] == 20
    path = config.config_dir() / "config.toml"
    path.write_text("# mine\nfps = 12\n")
    cfg, _ = config.load_config()
    assert cfg["fps"] == 12 and cfg["start"] == "random"      # defaults fill the rest
    assert path.read_text() == "# mine\nfps = 12\n"


def test_a_broken_config_falls_back_without_being_touched():
    path = config.config_dir() / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text("fps = = 3\n")
    cfg, problem = config.load_config()
    assert problem and cfg["fps"] == 30
    assert path.read_text() == "fps = = 3\n"


def test_presets_are_seeded_and_survive_a_round_trip():
    presets, problem = config.load_presets()
    assert problem is None and [p["name"] for p in presets] == [p["name"] for p in config.STARTERS]
    presets.append({"name": 'odd "name" é', "bloom": 0.35, "lines": ["radii", "flux"],
                    "hud": True, "line_width": 3})
    config.save_presets(presets)
    again, _ = config.load_presets()
    assert again == presets


def test_an_unreadable_presets_file_is_never_overwritten():
    path = config.presets_path()
    path.parent.mkdir(parents=True)
    path.write_text("[[preset]\n")
    presets, problem = config.load_presets()
    assert problem and presets
    assert path.read_text() == "[[preset]\n"


def test_deleted_presets_are_kept():
    config.archive_deleted({"name": "one", "bloom": 0.5})
    config.archive_deleted({"name": "two"})
    kept = tomllib.loads((config.config_dir() / "deleted-presets.toml").read_text())
    assert [p["name"] for p in kept["preset"]] == ["one", "two"]


def test_choose_and_unique_name():
    presets = [{"name": "a"}, {"name": "look 3"}]
    assert config.choose(presets, "first") == 0
    assert config.choose(presets, "look 3") == 1
    assert config.choose(presets, "none") is None
    assert config.choose(presets, "missing") is None
    assert config.choose([], "random") is None
    assert config.unique_name(presets) == "look 4"
