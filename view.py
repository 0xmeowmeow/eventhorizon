#!/usr/bin/env python
"""Render a black hole and either show it in a window or save it to a file.

Examples:
    python view.py                             # open a window
    python view.py --incl 0.6 --outer-edge 60  # a less edge-on view
    python view.py --plot isoradials           # lines of constant radius
    python view.py -o bh.png                   # write a file instead
"""

import argparse

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mass", type=float, default=1.0, help="black hole mass (natural units, G=c=1)")
    parser.add_argument("--incl", type=float, default=1.4,
                        help="observer inclination in radians: 0 is face-on, pi/2 is edge-on")
    parser.add_argument("--acc", type=float, default=1.0, help="accretion rate")
    parser.add_argument("--outer-edge", type=float, default=40.0, help="outer edge of the accretion disk")
    parser.add_argument("--plot", default="image",
                        choices=["image", "isoradials", "isoredshifts", "isofluxlines"],
                        help="what to draw. Default is the black hole image")
    parser.add_argument("-o", "--output", help="save to this file instead of opening a window")
    parser.add_argument("--dpi", type=int, default=120, help="resolution when saving")
    args = parser.parse_args()

    # Pick the backend before pyplot is imported: Agg renders without a display.
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from luminet.black_hole import BlackHole

    bh = BlackHole(mass=args.mass, incl=args.incl, acc=args.acc, outer_edge=args.outer_edge)

    if args.plot == "image":
        bh.plot()
    elif args.plot == "isoradials":
        bh.plot_isoradials(direct_r=range(6, int(args.outer_edge) + 1, 2), ghost_r=[])
    elif args.plot == "isoredshifts":
        bh.plot_isoredshifts()
    else:
        bh.plot_isofluxlines()

    if args.output:
        plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.output}")
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
