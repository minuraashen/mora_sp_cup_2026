"""make_architecture_fig.py -- the architecture diagram, in PNG, SVG and PDF.

    python scripts/tools/make_architecture_fig.py

Writes scripts/report/figs/architecture.{png,svg,pdf}: vector in SVG and PDF,
300 dpi in PNG.

Three panels:
  (a) the system -- what is measured, what is learned, what ships
  (b) the restoration core -- the NAFNet U-Net
  (c) one NAFBlock

Every label is read off the code, not off memory:
  widths / block counts   spcup/nafnet.py     ARCHS["w32"], NAFBlock.forward
  pipeline order          spcup/pipeline.py   denoise_image
  OOM ladder              spcup/infer.py      forward_with_oom_ladder
  noise model             train/noise_model.py

Drawing units are tenths of an inch, y up, and every coordinate below is explicit:
the layout is hand-packed, so if you add a box, check the panel bounds by eye.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

# --- palette: muted, print-safe, still separable in greyscale ---------------
C_CLASS = "#2C5F8A"   # classical signal processing
C_NET = "#B4581F"     # learned
C_IO = "#4A4A4A"      # tensors and files
C_TRAIN = "#3F7A4D"   # training time only
C_FRAME = "#C4C4C4"

FACE = {C_CLASS: "#E7EFF6", C_NET: "#F8E9DE", C_IO: "#F1F1F1", C_TRAIN: "#E5F0E8"}

UNITS = 10.0                      # drawing units per inch
X0, X1 = 0.0, 140.0               # canvas
Y0, Y1 = 4.0, 150.0
FIG_W = (X1 - X0) / UNITS
FIG_H = (Y1 - Y0) / UNITS

DASH_T = (0, (4, 2.5))            # training-time flow
DASH_S = (0, (1.4, 2.0))          # skip connection


class Box:
    __slots__ = ("cx", "cy", "w", "h")

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h

    @property
    def left(self):   return self.cx - self.w / 2

    @property
    def right(self):  return self.cx + self.w / 2

    @property
    def top(self):    return self.cy + self.h / 2

    @property
    def bottom(self): return self.cy - self.h / 2


def _step(fs):
    """Baseline-to-baseline distance, in drawing units, for a given point size."""
    return fs * 1.5 / 72.0 * UNITS


def box(ax, cx, cy, w, h, lines, color=C_IO, bold_first=False, lw=1.1):
    """Rounded box with centred multi-line text; lines are (text, fontsize) pairs."""
    b = Box(cx, cy, w, h)
    ax.add_patch(FancyBboxPatch(
        (b.left, b.bottom), w, h,
        boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=lw, edgecolor=color, facecolor=FACE.get(color, "#F2F2F2"),
        mutation_aspect=1.0, zorder=3))

    if isinstance(lines, str):
        lines = [(lines, 7.4)]
    total = sum(_step(fs) for _, fs in lines)
    top = cy + total / 2
    for i, (txt, fs) in enumerate(lines):
        s = _step(fs)
        ax.text(cx, top - s / 2, txt, ha="center", va="center", fontsize=fs,
                color="#111111", zorder=4,
                fontweight="bold" if (bold_first and i == 0) else "normal")
        top -= s
    return b


def arrow(ax, p0, p1, color=C_IO, style="-", lw=1.25):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=color, linestyle=style, shrinkA=0, shrinkB=0, zorder=2))


def arrow_lr(ax, a: Box, b: Box, **kw):
    arrow(ax, (a.right, a.cy), (b.left, b.cy), **kw)


def elbow(ax, pts, color=C_IO, style="-", lw=1.25):
    """Orthogonal polyline; the arrow head sits on the final segment."""
    for p0, p1 in zip(pts[:-1], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw,
                linestyle=style, solid_capstyle="round", zorder=2)
    arrow(ax, pts[-2], pts[-1], color=color, style=style, lw=lw)


def label(ax, x, y, txt, fs=6.6, color=C_IO, ha="center", va="center"):
    ax.text(x, y, txt, fontsize=fs, color=color, ha=ha, va=va, zorder=5,
            style="italic")


def panel(ax, x0, y0, x1, y1, title):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0,
                           edgecolor=C_FRAME, facecolor="#FCFCFC", zorder=0))
    ax.text(x0 + 2.2, y1 - 3.0, title, fontsize=10.0, fontweight="bold",
            color="#111111", ha="left", va="center", zorder=5)


def op(ax, cx, cy, sym, color=C_NET, r=1.5, fs=8):
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color,
                        linewidth=1.1, zorder=4))
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, zorder=5)
    return Box(cx, cy, 2 * r, 2 * r)


def row(ax, spec, y, x_left, x_right, gap, h):
    """Lay a list of (name, width, lines, color, lw) boxes out on one centred row."""
    total = sum(s[1] for s in spec) + gap * (len(spec) - 1)
    x = x_left + (x_right - x_left - total) / 2
    out = {}
    for name, w, lines, color, lw in spec:
        out[name] = box(ax, x + w / 2, y, w, h, lines, color, lw=lw)
        x += w + gap
    return out


# ===========================================================================
# (a) system
# ===========================================================================
def panel_a(ax):
    panel(ax, 2, 96, 138, 148,
          "(a)  System:  measured classical model (blue)  →  learned core (orange);"
          "   green runs at training time only")

    inp = box(ax, 17, 124, 24, 10,
              [("noisy sRGB", 8.4), ("992 × 992 × 3, uint8", 7.0)],
              C_IO, bold_first=True)
    core = box(ax, 50, 124, 28, 12,
               [("NAFNet-w32", 9.2), ("residual restoration core", 7.0),
                ("ŷ = x + fθ(x),  29.16 M params", 7.0)],
               C_NET, bold_first=True, lw=1.6)
    qnt = box(ax, 83, 124, 24, 10,
              [("clamp to [0, 1]", 8.4), ("quantise ⌊255x + ½⌋", 7.2)],
              C_CLASS, bold_first=True)
    out = box(ax, 114, 124, 24, 10,
              [("denoised PNG", 8.4), ("<id>.png", 7.0)], C_IO, bold_first=True)

    arrow_lr(ax, inp, core)
    arrow_lr(ax, core, qnt)
    arrow_lr(ax, qnt, out)
    label(ax, (inp.right + core.left) / 2, core.cy + 2.0, "x", 7.4)
    label(ax, (core.right + qnt.left) / 2, core.cy + 2.0, "ŷ", 7.4)
    label(ax, inp.cx, inp.top + 2.6, "9.7 s / image on one CPU", 6.8)

    ladder = box(ax, 74, 138, 78, 9,
                 [("robustness ladder  —  on OOM or any exception", 8.2),
                  ("whole image → tile 512 → tile 256 → classical fallback"
                   "  (gated adaptive median + σ-adaptive wavelet)", 6.9)],
                 C_CLASS, bold_first=True)
    elbow(ax, [(ladder.cx - 24, ladder.bottom), (ladder.cx - 24, 133.0),
               (core.cx, 133.0), (core.cx, core.top)], color=C_CLASS, style=DASH_T)

    meas = box(ax, 22, 107, 34, 13,
               [("noise characterisation", 8.4), ("460 public pairs", 6.9),
                ("var(I) = aI + b,  per channel", 6.9),
                ("excess kurtosis 1.86 – 2.36", 6.9),
                ("492 fixed hot pixels", 6.9)], C_CLASS, bold_first=True)
    gen = box(ax, 68, 107, 38, 13,
              [("Poisson–Gaussian generator", 8.4),
               ("severity s ~ U(l, h) with E[s²] = 1", 6.9),
               ("scale mixture: E[t] = 1, E[t²] = (K+3)/3", 6.9),
               ("+ impulses (0.033%) + 492 hot pixels", 6.9)], C_TRAIN, bold_first=True)
    trn = box(ax, 113, 107, 36, 13,
              [("fine-tuning", 8.4), ("init: released NAFNet-SIDD-w32", 6.9),
               ("PSNRLoss, AdamW, cosine LR, EMA", 6.9),
               ("batches: 60% real / 40% synthetic", 6.9)], C_TRAIN, bold_first=True)

    arrow_lr(ax, meas, gen, color=C_TRAIN, style=DASH_T)
    arrow_lr(ax, gen, trn, color=C_TRAIN, style=DASH_T)
    label(ax, (meas.right + gen.left) / 2, meas.cy + 2.0, "a, b, K", 6.8, C_TRAIN)
    label(ax, (gen.right + trn.left) / 2, meas.cy + 2.0, "(x̃, y)", 6.8, C_TRAIN)
    elbow(ax, [(trn.cx, trn.top), (trn.cx, 116.5), (core.cx, 116.5),
               (core.cx, core.bottom)], color=C_TRAIN, style=DASH_T)
    label(ax, trn.cx + 1.6, 114.0, "θ", 7.6, C_TRAIN, ha="left")

    label(ax, 4.5, 99.0,
          "Offline: the generator and the optimiser see the 414 training ids only; "
          "the 46 validation ids are held out end to end.",
          6.6, C_TRAIN, ha="left")


# ===========================================================================
# (b) NAFNet backbone
# ===========================================================================
def panel_b(ax):
    panel(ax, 2, 46, 138, 94,
          "(b)  Restoration core:  four-level NAFNet U-Net, additive skips, "
          "no padding at 992²")

    ye, yd = 82.0, 64.0
    enc_spec = [
        ("intro", 10.5, [("3 → 32", 6.9), ("conv 3×3", 6.9)], C_IO, 1.1),
        ("e1", 14.0, [("2× NAFBlock", 7.2), ("32 ch, 992²", 6.7)], C_NET, 1.1),
        ("d1", 8.0,  [("↓2", 7.2), ("conv s2", 6.4)], C_IO, 1.1),
        ("e2", 14.0, [("2× NAFBlock", 7.2), ("64 ch, 496²", 6.7)], C_NET, 1.1),
        ("d2", 8.0,  [("↓2", 7.2)], C_IO, 1.1),
        ("e3", 14.0, [("4× NAFBlock", 7.2), ("128 ch, 248²", 6.7)], C_NET, 1.1),
        ("d3", 8.0,  [("↓2", 7.2)], C_IO, 1.1),
        ("e4", 14.0, [("8× NAFBlock", 7.2), ("256 ch, 124²", 6.7)], C_NET, 1.1),
        ("d4", 8.0,  [("↓2", 7.2)], C_IO, 1.1),
        ("mb", 15.5, [("12× NAFBlock", 7.2), ("512 ch, 62²", 6.8)], C_NET, 1.7),
    ]
    enc = row(ax, enc_spec, ye, 5.0, 135.0, 1.7, 9.0)
    names = [s[0] for s in enc_spec]
    for a, b in zip(names[:-1], names[1:]):
        arrow_lr(ax, enc[a], enc[b])

    dec_spec = [
        ("u4", "d4", 8.0,  [("↑2", 7.2), ("pixshuf", 6.2)], C_IO),
        ("v4", "e4", 14.0, [("2× NAFBlock", 7.2), ("256 ch", 6.7)], C_NET),
        ("u3", "d3", 8.0,  [("↑2", 7.2)], C_IO),
        ("v3", "e3", 14.0, [("2× NAFBlock", 7.2), ("128 ch", 6.7)], C_NET),
        ("u2", "d2", 8.0,  [("↑2", 7.2)], C_IO),
        ("v2", "e2", 14.0, [("2× NAFBlock", 7.2), ("64 ch", 6.7)], C_NET),
        ("u1", "d1", 8.0,  [("↑2", 7.2)], C_IO),
        ("v1", "e1", 14.0, [("2× NAFBlock", 7.2), ("32 ch", 6.7)], C_NET),
        ("end", "intro", 10.5, [("32 → 3", 6.9), ("conv 3×3", 6.9)], C_IO),
    ]
    dec = {n: box(ax, enc[u].cx, yd, w, 9.0, lines, col)
           for n, u, w, lines, col in dec_spec}

    # middle block down the right-hand side into the first upsampler
    elbow(ax, [(enc["mb"].cx, enc["mb"].bottom), (enc["mb"].cx, yd),
               (dec["u4"].right, yd)])
    dnames = [s[0] for s in dec_spec]
    for a, b in zip(dnames[:-1], dnames[1:]):
        arrow(ax, (dec[a].left, yd), (dec[b].right, yd))

    for e, d in (("e4", "v4"), ("e3", "v3"), ("e2", "v2"), ("e1", "v1")):
        arrow(ax, (enc[e].cx, enc[e].bottom), (dec[d].cx, dec[d].top + 3.0),
              color=C_NET, style=DASH_S)
        op(ax, dec[d].cx, dec[d].top + 1.5, "+")
    label(ax, enc["d2"].cx, (ye + yd) / 2,
          "skips add,\nthey do not concatenate", 6.6, C_NET)

    # global residual: x is added back after the final 3x3 conv
    gsum = op(ax, dec["end"].cx, 54.0, "+", r=1.8, fs=9)
    arrow(ax, (dec["end"].cx, dec["end"].bottom), (gsum.cx, gsum.top))
    xline = enc["intro"].left - 3.0
    elbow(ax, [(enc["intro"].left, ye), (xline, ye), (xline, gsum.cy),
               (gsum.left, gsum.cy)])
    label(ax, xline + 1.2, (ye + yd) / 2, "x", 7.2, ha="left")

    bout = box(ax, gsum.cx + 23.0, 54.0, 32, 7.0,
               [("ŷ  →  clamp + quantise  →  PNG", 7.2)], C_IO)
    arrow(ax, (gsum.right, gsum.cy), (bout.left, bout.cy))

    label(ax, 4.5, 49.0,
          "992 = 2⁵ · 31, so a four-level U-Net needs zero padding; peak working "
          "set 1.28 GB, so the whole image runs in a single pass and tiling is only a "
          "fallback.", 6.6, ha="left")


# ===========================================================================
# (c) NAFBlock
# ===========================================================================
def panel_c(ax):
    panel(ax, 2, 8, 138, 44,
          "(c)  NAFBlock:  activation-free, two residual branches (C channels in, C out)")

    yb = 26.0
    spec = [
        ("in",  7.0,  [("in", 7.4)], C_IO, 1.1),
        ("ln1", 8.6,  [("LayerNorm", 6.6)], C_IO, 1.1),
        ("c1",  8.0,  [("1×1", 6.9), ("C → 2C", 6.6)], C_NET, 1.1),
        ("dw",  8.0,  [("DWConv", 6.9), ("3×3", 6.6)], C_NET, 1.1),
        ("sg1", 10.2, [("SimpleGate", 6.9), ("2C → C", 6.6)], C_NET, 1.1),
        ("mul", 3.6,  None, None, None),
        ("c3",  8.0,  [("1×1", 6.9), ("C → C", 6.6)], C_NET, 1.1),
        ("a1",  4.0,  None, None, None),
        ("ln2", 8.6,  [("LayerNorm", 6.6)], C_IO, 1.1),
        ("c4",  8.0,  [("1×1", 6.9), ("C → 2C", 6.6)], C_NET, 1.1),
        ("sg2", 10.2, [("SimpleGate", 6.9), ("2C → C", 6.6)], C_NET, 1.1),
        ("c5",  8.0,  [("1×1", 6.9), ("C → C", 6.6)], C_NET, 1.1),
        ("a2",  4.0,  None, None, None),
        ("out", 7.0,  [("out", 7.4)], C_IO, 1.1),
    ]
    gap = 1.9
    total = sum(s[1] for s in spec) + gap * (len(spec) - 1)
    x = 5.0 + (133.0 - total) / 2
    nb = {}
    for name, w, lines, col, lw in spec:
        cx = x + w / 2
        if lines is None:
            nb[name] = op(ax, cx, yb, "×" if name == "mul" else "+",
                          r=w / 2, fs=9)
        else:
            nb[name] = box(ax, cx, yb, w, 8.0, lines, col, lw=lw)
        x += w + gap

    order = [s[0] for s in spec]
    for a, b in zip(order[:-1], order[1:]):
        arrow(ax, (nb[a].right, yb), (nb[b].left, yb))
    label(ax, (nb["c3"].right + nb["a1"].left) / 2, yb + 2.4, "β", 7.2, C_NET)
    label(ax, (nb["c5"].right + nb["a2"].left) / 2, yb + 2.4, "γ", 7.2, C_NET)

    sca = box(ax, (nb["sg1"].cx + nb["mul"].cx) / 2, 16.0, 30, 7.0,
              [("SCA:  global avg-pool → 1×1 conv", 7.0)], C_NET)
    elbow(ax, [(nb["sg1"].cx, nb["sg1"].bottom), (nb["sg1"].cx, sca.cy),
               (sca.left, sca.cy)])
    elbow(ax, [(sca.right, sca.cy), (nb["mul"].cx, sca.cy),
               (nb["mul"].cx, nb["mul"].bottom)])

    for src, dst, ytop in ((nb["in"], nb["a1"], 33.0), (nb["a1"], nb["a2"], 37.0)):
        elbow(ax, [(src.cx, src.top), (src.cx, ytop), (dst.cx, ytop),
                   (dst.cx, dst.top)], color=C_NET)

    label(ax, 4.5, 11.0,
          "No ReLU or GELU anywhere: SimpleGate (x₁ ⊙ x₂ on a channel split) "
          "and simplified channel attention replace them — the reason the core stays "
          "affordable on CPU.", 6.6, ha="left")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "report", "figs"))
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["svg.fonttype"] = "path"     # SVG renders identically without the font
    plt.rcParams["pdf.fonttype"] = 42         # embed TrueType, not Type 3

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    panel_a(ax)
    panel_b(ax)
    panel_c(ax)

    for ext, kw in (("png", dict(dpi=300)), ("svg", {}), ("pdf", {})):
        path = os.path.join(outdir, f"architecture.{ext}")
        fig.savefig(path, facecolor="white", **kw)
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
