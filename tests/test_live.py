from luminet.live import keys_in


def test_keys_in_keeps_plain_keys_in_order():
    assert list(keys_in("BBkq")) == ["B", "B", "k", "q"]


def test_keys_in_drops_arrow_and_function_keys():
    # Up, down, right, left, then F1 in its SS3 form and F5 as a CSI sequence.
    assert list(keys_in("\x1b[A\x1b[B\x1b[C\x1b[DxOP\x1bOP\x1b[15~y")) == ["x", "O", "P", "y"]


def test_keys_in_keeps_a_lone_escape():
    assert list(keys_in("\x1b")) == ["\x1b"]
