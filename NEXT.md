# Where this was left (parked 2026-09-17)

Branch `fix/photon-sampling-rng`; `master` is still upstream. Nothing uncommitted.

## Running it

    luminet spin            # plot1979, ink palette, live; h for keys, q to quit

`luminet` is an alias in ~/.zshrc for `luminet-launcher`, which uses the
project venv. numba is installed there; the status line says `compiled` when
the fast path is active.

## Open decisions

- **Frame rate.** Settled: 30fps is the default and costs about a third of a
  core at full screen, since brightness per cell is now measured once per map
  and window instead of every frame.
- **Ghostty glow.** A custom shader post-processing what the app draws. Parked.
- **Pixel mode (x, --pixels) is unverified on a real screen.** Checked only by
  decoding what the app sends. Whether Ghostty accepts the /dev/shm file route
  is found by probing at startup; if not it falls back to half resolution sent
  inline. Full screen costs about 55% of a core at 30fps, against 34% for
  braille. Dots are sparser than the plate; --pixel-grain and --ink-gamma tune it.

- **Inclination and zoom in real time.** Discussed, not built. Tilting now
  re-solves the lensing map and rebuilds the dot field, about 1-3 seconds a
  step. Real time would mean precomputing maps across a range of inclinations
  once, in a subprocess, and blending between neighbours. Zoom could resample
  the per-cell brightness while the gesture lasts and re-measure after.

## Known rough edges

- `--infall` shifts gas between discrete rings rather than moving it smoothly.
- Octants (Unicode 16) are not implemented; Python 3.13 cannot name them.
- True pixel-sized stars or dots would need the kitty graphics protocol.
- Measuring CPU through a pseudo-terminal: pass the interpreter's absolute path
  as argv[0], or Python resolves to the system interpreter and loses the venv.

## Elsewhere

- `../plant/canalisation` is cloned and parked, untouched.
- A `tuilab` alias was added to ~/.zshrc for the tui-research project.
