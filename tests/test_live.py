from luminet.live import keys_in


def test_keys_in_keeps_plain_keys_in_order():
    assert list(keys_in("BBkq")) == ["B", "B", "k", "q"]


def test_keys_in_drops_arrow_and_function_keys():
    # Up, down, right, left, then F1 in its SS3 form and F5 as a CSI sequence.
    assert list(keys_in("\x1b[A\x1b[B\x1b[C\x1b[DxOP\x1bOP\x1b[15~y")) == ["x", "O", "P", "y"]


def test_keys_in_keeps_a_lone_escape():
    assert list(keys_in("\x1b")) == ["\x1b"]


def test_keys_in_reports_focus_and_keeps_tab():
    from luminet.live import FOCUS_IN, FOCUS_OUT

    assert list(keys_in("\x1b[Oa\x1b[I\t")) == [FOCUS_OUT, "a", FOCUS_IN, "\t"]


def test_typed_options_are_told_apart_from_defaults():
    from luminet import cli

    parser = cli.parser_for_spin()
    given = cli.explicit_options(parser, ["--palette=ice", "--bloom", "0.4", "--hud"])
    assert {"palette", "bloom", "hud"} <= given
    assert "speed" not in given and "incl" not in given


def make_live(tmp_path, monkeypatch, argv=()):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from luminet import cli, live

    parser = cli.parser_for_spin()
    args = parser.parse_args(list(argv))
    cli.widget_settings(args, parser, list(argv))
    settings = {**cli.DEFAULTS}
    return live.Live(settings, args)


def test_a_saved_look_restores_exactly(tmp_path, monkeypatch):
    first = make_live(tmp_path, monkeypatch, ["--preset", "none"])
    first.palette_at, first.glow_at, first.bloom = 5, 3, 0.75
    first.families = {"radii", "flux"}
    first.line_style, first.line_colour, first.line_width = 2, 4, 5
    first.vignette_on, first.mask_on, first.effects.hud = True, False, True
    first.settings.update(incl=1.1, mass=2.0, outer_edge=64.0)
    first.speed, first.dust = 0.3, 0.02
    first.save_preset()

    second = make_live(tmp_path, monkeypatch, ["--preset", first.presets[-1]["name"]])
    assert second.current_look() == first.current_look()


def test_command_line_options_win_over_the_opening_preset(tmp_path, monkeypatch):
    shown = make_live(tmp_path, monkeypatch, ["--preset", "ember", "--palette", "ice"])
    look = shown.current_look()
    assert look["palette"] == "ice"            # typed
    assert look["bloom"] == 0.9                # from the preset


def test_deleting_a_preset_takes_two_presses(tmp_path, monkeypatch):
    shown = make_live(tmp_path, monkeypatch, ["--preset", "first"])
    count = len(shown.presets)
    shown.handle("X")
    assert len(shown.presets) == count
    shown.handle("X")
    assert len(shown.presets) == count - 1
    from luminet import config

    assert len(config.load_presets()[0]) == count - 1
    assert (config.config_dir() / "deleted-presets.toml").exists()
