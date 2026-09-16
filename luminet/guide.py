"""Plain explanations of what the controls do.

Each entry says what the control is in the physical system, what actually changes
on screen when you move it, and a comparison that shows the effect. Nothing here
is withheld or gated: `luminet explain` prints the lot.
"""

ORIENTATION = """\
WHAT YOU ARE LOOKING AT

  A black hole with a thin disk of hot gas orbiting it. Start from something
  easier: Saturn and its rings. A flat bright ring, seen from some angle.

  If light travelled in straight lines, that is all this would be. Tilt the ring
  and it squashes into an ellipse; the far half passes behind the hole and is
  hidden, exactly as Saturn's far rings go behind Saturn.

  Light does not travel in straight lines here. The far half of the disk is not
  hidden: its light bends up and over the hole and reaches you anyway, so the
  back of the disk appears lifted above the middle. A second, flatter image of
  the disk's underside appears below. Everything beyond the plain ellipse is
  light that was bent on the way to you.

  You can see this directly. These two draw the same disk, the first with
  gravity switched off, the second with it on:

      luminet render --plot flat
      luminet render --plot lines

  The dark middle is not the hole. It is the shadow: the directions in which
  every line of sight ends on the hole. It is noticeably bigger than the hole.

  Brightness is not even, for two reasons worth keeping separate. Gas coming
  towards you is beamed brighter and shifted bluer; gas going away is dimmed and
  reddened. Separately, all light loses energy climbing out of the gravity well,
  most of all near the middle.
"""

PARAMS = {
    "plot": {
        "what": "Which view to draw. Five of them.",
        "watch": """\
  image         the disk as a camera would record it, shaded by brightness
  lines         the classic line drawing: a few rings, direct image and ghost
  flat          the same rings with gravity switched off, for comparison
  isoradials    rings of constant distance from the hole, shaded by brightness
  isoredshifts  lines of constant Doppler/gravitational shift
  isofluxlines  contours of constant brightness""",
        "try": "luminet sweep --plot flat,lines,image",
    },
    "incl": {
        "what": """\
  Where you are, relative to the disk, in radians. 0 is directly above, looking
  down on the ring face-on. 1.571 is pi/2, in the disk's own plane, edge-on.""",
        "watch": """\
  Two different things happen as this grows, and they are easy to confuse. The
  ring squashes into an ellipse, which is only perspective and needs no gravity.
  And the far side peels up over the top of the hole, which is entirely gravity.
  Above about 1.0 the ghost image separates out below.

  At 0 the ring is evenly bright: everything moves across your line of sight, so
  nothing is beamed towards you. Tilt it and one side brightens sharply.""",
        "try": "luminet sweep --incl 0.1,0.5,0.9,1.3,1.5",
    },
    "mass": {
        "what": """\
  Sets the scale of everything, in natural units where G = c = 1. The disk's
  inner edge is always 6 x mass, the closest a stable orbit can get. The photon
  sphere sits at about 5.2 x mass.""",
        "watch": """\
  On its own, less than you would expect. Mass and outer_edge only matter as a
  ratio: the disk's width measured in black hole radii. Doubling both together
  produces a byte-identical image, so mass alone is a zoom, not a new shape.
  Changing mass while outer_edge stays put does change the picture, but what
  changed was really the ratio.""",
        "try": "luminet sweep --mass 1,2,4   # then add --outer-edge and scale it too",
    },
    "acc": {
        "what": "Accretion rate: how much matter falls in per unit time. Scales the emitted flux linearly.",
        "watch": """\
  Nothing at all, in the rendered image. The picture is normalised to its own
  brightest point, so multiplying acc by ten gives a byte-identical file. It is
  a real physical quantity and it does change the numbers, but not the view.
  To see it do something, look at the flux column instead:

      luminet photons -n 5 --seed 1 --acc 1
      luminet photons -n 5 --seed 1 --acc 10""",
        "try": "luminet photons -n 5 --seed 1 --acc 10",
    },
    "outer_edge": {
        "what": "Where the disk stops, in the same units as mass. The inner edge is not adjustable; it is fixed by the physics at 6 x mass.",
        "watch": """\
  A wider disk adds material far out, which is dimmer, slower and barely bent.
  The bright lensed structure near the middle stays where it is. Narrow it
  towards 6 and you are left with only the strongly curved part.""",
        "try": "luminet sweep --outer-edge 10,20,40,100",
    },
    "resolution": {
        "what": "How many rings are drawn, and how many points around each ring.",
        "watch": """\
  Smoothness and render time, nothing else. No physics changes. Low values make
  curves visibly polygonal. This is the dial to turn down while exploring and up
  for a final picture.""",
        "try": "luminet sweep --resolution 30,80,200",
    },
    "color_by": {
        "what": """\
  What the shading means. 'flux' is how bright the gas appears: what a camera
  records. 'redshift' is whether light arrived with more or less energy than it
  left with.""",
        "watch": """\
  In redshift, blue is the side rotating towards you and red the side going
  away, plus a general reddening near the middle from climbing out of the well.
  It looks washed out because that central gravitational redshift is extreme and
  sets the colour scale for everything else.""",
        "try": "luminet sweep --color-by flux,redshift --incl 1.0",
    },
    "cmap": {
        "what": "The matplotlib colour palette. Purely presentation.",
        "watch": "No physics whatsoever. Greys_r is the default; inferno and magma look more like the usual published images.",
        "try": "luminet sweep --cmap Greys_r,inferno,magma",
    },
    "radii": {
        "what": "For --plot lines and flat: which rings of the disk to draw, as distances from the hole.",
        "watch": "Values below 6 are refused: no stable orbit exists there, so there is no gas to draw.",
        "try": "luminet render --plot lines --radii 6,7,9,12,20,35",
    },
    "ghost_radii": {
        "what": "For --plot lines: which rings to draw of the second image, the one whose light bends right around the hole.",
        "watch": """\
  These may exceed outer_edge, and usually should. Light from material far
  outside the disk still wraps around into view, drawing the tight nested curves
  in the lower half. Set this empty to see how much of the figure is ghost.""",
        "try": "luminet render --plot lines --ghost-radii ''",
    },
    "line_color": {
        "what": "Colour of the lines in --plot lines and flat.",
        "watch": "Presentation only. Takes any matplotlib colour, including hex like '#66ccff'.",
        "try": "luminet render --plot lines --line-color '#66ccff'",
    },
    "lw": {
        "what": "Line width for --plot lines and flat.",
        "watch": "Presentation only.",
        "try": "luminet render --plot lines --lw 2.5",
    },
}

ORDER = ["plot", "incl", "outer_edge", "mass", "acc", "resolution",
         "color_by", "cmap", "radii", "ghost_radii", "line_color", "lw"]

# A one-line version for the interactive prompts, so the menu is not a list of
# bare names with defaults attached.
SHORT = {
    "plot": "which view to draw",
    "incl": "viewing angle, 0 looks down on the disk, 1.571 is edge-on",
    "mass": "scale; only the ratio to outer_edge changes the shape",
    "acc": "accretion rate; does not change the image, only flux numbers",
    "outer_edge": "where the disk stops; inner edge is fixed at 6 x mass",
    "resolution": "smoothness and render time, no physics",
    "color_by": "flux is what a camera sees, redshift is energy gained or lost",
    "cmap": "colour palette, presentation only",
    "radii": "which rings to draw",
    "ghost_radii": "which rings of the bent-around second image to draw",
    "line_color": "colour of the lines",
    "lw": "line width",
}


def explain(name=None):
    """Return the explanation for one control, or the whole guide."""
    if name is None:
        out = [ORIENTATION, "THE CONTROLS\n"]
        for key in ORDER:
            out.append(_one(key))
        out.append(
            "  Every render is recorded. `luminet log` lists what you have run,\n"
            "  `luminet vary <n> --incl 0.9` repeats one with a single change and\n"
            "  shows it beside the original, and `luminet note <n> \"...\"` keeps an\n"
            "  observation with it.\n"
        )
        return "\n".join(out)

    key = name.replace("-", "_")
    if key not in PARAMS:
        known = ", ".join(ORDER)
        return f"No such control: {name}\nKnown controls: {known}"
    return _one(key)


def _one(key):
    p = PARAMS[key]
    lines = [f"  {key}", f"    {SHORT[key]}", "", _indent(p["what"]), "",
             "    what changes:", _indent(p["watch"]), "",
             f"    try:  {p['try']}", ""]
    return "\n".join(lines)


def _indent(text):
    return "\n".join(
        ("    " + line.strip()) if not line.startswith("  ") else ("  " + line)
        for line in text.rstrip().splitlines()
    )
