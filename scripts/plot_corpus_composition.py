"""Regenerate Figure 8 (corpus composition) of the Mind2Dialogue paper.

Counts, colours, explode offsets and wedge geometry were recovered exactly from
the previous PDFs; both panels sum to 6,330 conversations.  The centre
annotation that used to read "6,330 conversations" / "6,330 items" is gone.
"""
import json, math, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.load(open(HERE / "corpus_composition_data.json"))
TEXT = "#2b2b2b"
LEAD = "#9a9a9a"

def donut(ax, counts, colors, explode, labels, title, pct_floor=3.0):
    total = sum(counts)
    wedges, _ = ax.pie(
        counts, colors=colors, explode=explode,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    ax.set_title(title, fontsize=18, fontweight="bold", color=TEXT, pad=14)

    for w, c in zip(wedges, counts):
        pct = 100.0 * c / total
        if pct < pct_floor:
            continue
        a = math.radians((w.theta1 + w.theta2) / 2)
        ax.text(w.center[0] + 0.79 * math.cos(a), w.center[1] + 0.79 * math.sin(a),
                f"{pct:.0f}%", ha="center", va="center",
                fontsize=14.5, fontweight="bold", color="white", zorder=6)

    named = []
    for w, c, l in zip(wedges, counts, labels):
        if not l:
            continue
        a = math.radians((w.theta1 + w.theta2) / 2)
        named.append((l, c, w.center[0] + math.cos(a), w.center[1] + math.sin(a)))

    right = [t for t in named if t[2] >= 0]
    left = [t for t in named if t[2] < 0]
    for side, group in (("right", right), ("left", left)):
        if not group:
            continue
        group = sorted(group, key=lambda t: -t[3])          # top wedge -> top label
        n = len(group)
        step = 0.285
        span = step * (n - 1)
        top = min(span / 2, 1.18)
        ys = [top - i * (2 * top / (n - 1)) for i in range(n)] if n > 1 else [0.0]
        sgn = 1 if side == "right" else -1
        xt = sgn * 1.42
        for (l, c, x0, y0), y in zip(group, ys):
            xb = x0 + sgn * 0.30
            ax.plot([xt, xb, x0], [y, y, y0], color=LEAD, lw=0.8, zorder=4,
                    solid_capstyle="round", clip_on=False)
            ax.plot([x0], [y0], "o", ms=3.4, color=LEAD, zorder=5, clip_on=False)
            ax.text(xt + sgn * 0.06, y, f"{l}   {100.0 * c / total:.1f}%",
                    ha="left" if side == "right" else "right", va="center",
                    fontsize=13.8, fontweight="bold", color=TEXT, clip_on=False)
    ax.set_aspect("equal")
    ax.axis("off")

PAD = 0.06

def build(key, title):
    d = DATA[key]
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    donut(ax, d["counts"], [tuple(c) for c in d["colors"]], d["explode"], d["labels"], title)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(PAD)
    return fig, bb

if __name__ == "__main__":
    from matplotlib.transforms import Bbox
    # panel widths in the LaTeX figure; pad the shorter panel so both render
    # to the same height at those widths
    PANEL = {"scenario": 0.55, "specialization": 0.44}
    figs = {k: build(k, t) for k, t in
            [("scenario", "Scenario Categories"), ("specialization", "Specialization Clusters")]}
    tall = max(PANEL[k] * bb.height / bb.width for k, (_, bb) in figs.items())
    for k, (fig, bb) in figs.items():
        want = tall * bb.width / PANEL[k]
        extra = max(want - bb.height, 0.0) / 2
        box = Bbox.from_extents(bb.x0, bb.y0 - extra, bb.x1, bb.y1 + extra)
        out = HERE / f"donuts_{k}.pdf"
        fig.savefig(out, bbox_inches=box)
        plt.close(fig)
        print(f"wrote {out}  {box.width:.3f}x{box.height:.3f} in  "
              f"latex height = {PANEL[k] * box.height / box.width:.4f} linewidth")
