# Where this was left (parked 2026-09-17)

Branch `fix/photon-sampling-rng`; `master` is still upstream. Nothing uncommitted.

## Running it

    luminet spin            # plot1979, ink palette, live; h for keys, q to quit

`luminet` is an alias in ~/.zshrc for `luminet-launcher`, which uses the
project venv. numba is installed there; the status line says `compiled` when
the fast path is active.

## Open decisions

- **Frame rate default.** 20fps uses 65% of a core at full screen, 10fps 32%.
  The inner ring orbits every ~8s, so 10fps looks nearly as smooth. Proposed
  as the default for a desktop toy; not yet agreed.
- **Ghostty glow.** A custom shader that post-processes what the app draws
  (bloom, phosphor, CRT). Discussed as a later layer; Ghostty shaders get only
  the terminal image, time and size, so no physics can go there.

## Known rough edges

- `--infall` shifts gas between discrete rings rather than moving it smoothly.
- Octants (Unicode 16) are not implemented; Python 3.13 cannot name them.
- True pixel-sized stars or dots would need the kitty graphics protocol.
- Measuring CPU through a pseudo-terminal: pass the interpreter's absolute path
  as argv[0], or Python resolves to the system interpreter and loses the venv.

## Elsewhere

- `../plant/canalisation` is cloned and parked, untouched.
- A `tuilab` alias was added to ~/.zshrc for the tui-research project.
