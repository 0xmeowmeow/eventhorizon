# Where this was left (2026-09-18)

Branch `fix/photon-sampling-rng`; `master` is still upstream.

## Running it

    luminet spin            # the widget; h for keys, tab for status, q to quit

`luminet` is an alias in ~/.zshrc for `luminet-launcher`, which uses the
project venv (numba installed; status says `compiled`).

## v1 widget: built

- Presets in ~/.config/luminet/presets.toml (app-managed; five starters seeded
  on first run). 1-9 / 0 pick, + saves, X X deletes (archived to
  deleted-presets.toml). config.toml is written once and never rewritten.
  Precedence: typed options > opening preset > config > defaults.
- Pixels by default when the terminal answers a graphics probe (kitty query
  plus a DA1 sentinel), else braille. Images are placed at z=-1 so text can
  sit over them.
- Full window; the status line is hidden and drawn over the bottom row (tab).
- Focus reports (CSI ?1004h): 5fps while unfocused (config unfocused_fps, 0
  disables). Waiting is on select(), so keys answer at once at low rates.
- SIGHUP/SIGTERM exit cleanly: images deleted, /dev/shm files removed,
  terminal modes restored. Verified through a pty.
- First run with no bank: a notice explaining the bank, then a bottom-line
  count until it completes.
- `--version`, grouped `spin --help`, README leads with the widget.

## Events (luminet/effects.py): built

`!` probe, `@` cuneiform transmission, `#` HUD, `$` warp, `A` automatic events
(every 4-10 min, config). One canvas interface paints into both the pixel
image and the braille cells; glyphs go on as text. About 2 ms a frame with
everything running. Verified by decoding pty output, not yet by eye on a real
screen.

## Open

- **Name.** The user is choosing one; `luminet` is taken on PyPI. The config
  directory name lives in `config.APP`.
- **GitHub.** Agreed: fork bgmeulem/Luminet to 0xmeowmeow and push. A fork of a
  public repo is public. An upstream PR would carry only the library fixes (RNG
  sampling, redshift defaults, color_by redshift, cos_gamma warning), on its
  own branch.
- **Packaging** for Debian and Arch/Omarchy, after the name.
- The grey band just outside the shadow in pixel mode predates this work.
- Unfocused 5fps may be wrong for a desktop widget that is rarely focused;
  unfocused_fps = 0 in config.toml turns it off.
- Ghostty glow shader, octants, smooth infall: parked.

## Known rough edges

- `--infall` shifts gas between discrete rings rather than moving it smoothly.
- Measuring CPU through a pseudo-terminal: pass the interpreter's absolute path
  as argv[0], or Python resolves to the system interpreter and loses the venv.

## Elsewhere

- `../plant/canalisation` is cloned and parked, untouched.
- A `tuilab` alias was added to ~/.zshrc for the tui-research project.
