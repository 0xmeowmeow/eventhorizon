# Where this was left (2026-09-18)

Branch `fix/photon-sampling-rng`; `master` is still upstream.

## Running it

    eventhorizon            # the widget (also eh, or luminet spin); h for keys

`eventhorizon` and `eh` are aliases in ~/.zshrc for the project venv's
scripts; `luminet` still runs the simulation explorer. The distribution is
eventhorizon 1.0.0; the Python package inside is still `luminet`.

## v1 widget: built

- Presets in ~/.config/eventhorizon/presets.toml (app-managed; eleven
  starters: five made with the widget, six the user made while playing, their
  duplicates condensed). 1-9 / 0 / < > pick, + saves, X X deletes (archived to
  deleted-presets.toml). The user's file from before the rename is kept as
  presets-before-eventhorizon.toml. config.toml is written once and never rewritten.
  Precedence: typed options > opening preset > config > defaults.
- Pixels by default when the terminal answers a graphics probe (kitty query
  plus a DA1 sentinel), else braille. Images are placed at z=-1 so text can
  sit over them.
- Full window; the status line is hidden and drawn over the bottom row (tab).
- Focus reports (CSI ?1004h): 20fps while unfocused (5 looked laggy) (config unfocused_fps, 0
  disables). Waiting is on select(), so keys answer at once at low rates.
- SIGHUP/SIGTERM exit cleanly: images deleted, /dev/shm files removed,
  terminal modes restored. Verified through a pty.
- First run with no bank: a notice explaining the bank, then a bottom-line
  count until it completes.
- `--version`, grouped `spin --help`, README leads with the widget.

## Events (luminet/effects.py): built

`!` probe, `@` cuneiform transmission, `#` HUD, `$` warp out and, pressed
again, back (automatic warps return after 30s), `I` invert, `A` automatic events
(every 4-10 min, config). One canvas interface paints into both the pixel
image and the braille cells; glyphs go on as text. About 2 ms a frame with
everything running. Verified by decoding pty output, not yet by eye on a real
screen.

## Open

- **Name: eventhorizon**, commands eventhorizon and eh. `eventhorizon` and `eh`
  are both taken on PyPI (unrelated tools), so a PyPI release would need
  another distribution name; Debian and Arch names are free.
- **GitHub.** Agreed: fork bgmeulem/Luminet to 0xmeowmeow and push. A fork of a
  public repo is public. An upstream PR would carry only the library fixes (RNG
  sampling, redshift defaults, color_by redshift, cos_gamma warning), on its
  own branch.
- **Packaging** for Debian and Arch/Omarchy, after the name.
- The grey band just outside the shadow in pixel mode predates this work.
- Ghostty glow shader, octants, smooth infall: parked.

## Known rough edges

- `--infall` shifts gas between discrete rings rather than moving it smoothly.
- Measuring CPU through a pseudo-terminal: pass the interpreter's absolute path
  as argv[0], or Python resolves to the system interpreter and loses the venv.

## Elsewhere

- `../plant/canalisation` is cloned and parked, untouched.
- A `tuilab` alias was added to ~/.zshrc for the tui-research project.
