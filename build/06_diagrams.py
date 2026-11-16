"""Draw the notebook diagrams as PNG files in img/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

IMG = Path(__file__).resolve().parent.parent / "img"; IMG.mkdir(exist_ok=True)
NAVY, GREEN, AMBER, RED, SLATE, BLUE, INK = "#1F2A44", "#059669", "#B45309", "#B91C1C", "#64748B", "#2563EB", "#0F172A"
FILL = {NAVY: "#E2E8F0", GREEN: "#D1FAE5", AMBER: "#FEF3C7", RED: "#FEE2E2", BLUE: "#DBEAFE", SLATE: "#F1F5F9"}
plt.rcParams["font.family"] = "DejaVu Sans"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def box(ax, x, y, w, h, title, sub="", color=NAVY, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[color], ec=color, lw=1.8))
    ax.text(x + w / 2, y + h / 2 + (0.17 if sub else 0), title, ha="center", va="center",
            fontsize=size, fontweight="bold", color=color)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=9, color=INK, linespacing=1.3)


def arrow(ax, p, q, color=SLATE, label="", rad=0.0, dy=0.14, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, ha="center", va="center", fontsize=9, color=color,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)


def save(fig, name):
    fig.savefig(IMG / name, facecolor="white"); plt.close(fig)


# 1. Roadmap -----------------------------------------------------------------------------
fig, ax = canvas(11, 1.9)
parts = [("1  Prices", "a tool that\nreads numbers", BLUE, "5 min"), ("2  Embeddings", "a tool that\nreads text", BLUE, "10 min"),
         ("3  The agent", "a model that\nchooses the tools", AMBER, "12 min"), ("4  Guardrails", "limits on what\nthe agent may do", RED, "10 min")]
for i, (t, s, c, m) in enumerate(parts):
    x = 0.25 + i * 2.7
    box(ax, x, 0.45, 2.3, 1.2, t, s, c)
    ax.text(x + 1.15, 0.2, m, ha="center", fontsize=9, color=SLATE)
    if i < 3:
        arrow(ax, (x + 2.33, 1.05), (x + 2.67, 1.05))
save(fig, "roadmap.png")

# 2. Agent loop --------------------------------------------------------------------------
fig, ax = canvas(11, 3.9)
box(ax, 0.2, 1.75, 1.9, 1.0, "Question", "from the user", NAVY)
box(ax, 3.0, 1.65, 2.3, 1.2, "MODEL", "thinks, then picks\nthe next move", AMBER, 13)
box(ax, 8.2, 2.6, 2.5, 0.95, "TOOLS", "prices, search", BLUE)
box(ax, 8.2, 0.75, 2.5, 0.95, "Answer", "with sources", GREEN)
arrow(ax, (2.15, 2.25), (2.95, 2.25))
arrow(ax, (5.35, 2.75), (8.15, 3.3), AMBER, "1. ACT: call a tool", rad=-0.12, dy=0.42)
arrow(ax, (8.15, 2.85), (5.35, 2.3), BLUE, "2. OBSERVE: read the result", rad=-0.12, dy=-0.5)
arrow(ax, (5.35, 1.85), (8.15, 1.2), GREEN, "3. Enough? Then answer", dy=-0.3)
ax.text(4.15, 1.2, "repeat 1 and 2\nuntil the model has enough", ha="center", fontsize=9, color=SLATE, style="italic")
ax.text(9.45, 2.35, "ordinary code: fast, exact, auditable", fontsize=8.5, color=SLATE, ha="center")
ax.text(4.15, 3.1, "a language model: flexible, fallible", fontsize=8.5, color=SLATE, ha="center")
save(fig, "agent_loop.png")

# 3. Embeddings --------------------------------------------------------------------------
fig, ax = canvas(11, 3.0)
box(ax, 0.2, 1.55, 2.9, 0.75, '"The bank raised its dividend"', "", NAVY, 9.5)
box(ax, 0.2, 0.55, 2.9, 0.75, '"The lender increased its payout"', "", NAVY, 9.5)
box(ax, 3.9, 0.95, 1.9, 1.0, "Embedding\nmodel", "", AMBER, 11)
arrow(ax, (3.15, 1.9), (3.85, 1.6)); arrow(ax, (3.15, 0.95), (3.85, 1.3))
ax.text(6.3, 1.85, "[0.12, -0.03, 0.44, ...]", fontsize=10, color=INK, family="DejaVu Sans Mono")
ax.text(6.3, 1.0, "[0.10, -0.01, 0.47, ...]", fontsize=10, color=INK, family="DejaVu Sans Mono")
ax.text(7.25, 1.43, "384 numbers per text", fontsize=9, color=SLATE, ha="center")
arrow(ax, (5.85, 1.45), (6.25, 1.45))
rng = np.random.default_rng(3)
ax.add_patch(FancyBboxPatch((8.75, 0.3), 2.05, 2.35, boxstyle="round,pad=0.02,rounding_size=0.1", fc="white", ec=SLATE, lw=1))
for (cx, cy), c in [((9.25, 2.1), BLUE), ((10.3, 1.8), GREEN), ((9.6, 0.85), SLATE)]:
    pts = rng.normal(0, 0.13, (9, 2)) + [cx, cy]
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=c, alpha=.6)
ax.scatter([9.2, 9.32], [2.12, 2.02], s=70, color=RED, marker="*", zorder=3)
ax.text(9.78, 0.42, "close together =\nsimilar meaning", ha="center", fontsize=8.5, color=INK)
ax.text(9.78, 2.8, "map of meaning", ha="center", fontsize=9, color=SLATE)
arrow(ax, (8.3, 1.45), (8.7, 1.45))
save(fig, "embeddings.png")

# 4. Guardrail ---------------------------------------------------------------------------
fig, ax = canvas(11, 3.9)
box(ax, 0.2, 1.75, 2.2, 1.1, "MODEL", "proposes a move", AMBER, 13)
box(ax, 3.9, 1.75, 2.2, 1.1, "GUARDRAIL", "your rule, in code", RED, 13)
box(ax, 7.9, 2.75, 2.9, 0.9, "Read-only tools", "prices, search", BLUE)
box(ax, 7.9, 0.85, 2.9, 0.9, "Action tools", "email, orders, payments", NAVY)
arrow(ax, (2.45, 2.3), (3.85, 2.3))
arrow(ax, (6.15, 2.55), (7.85, 3.15), GREEN, "allowed", rad=-0.1, dy=0.3)
arrow(ax, (6.15, 2.05), (7.85, 1.35), GREEN, "allowed only if\nthe rule says yes", rad=0.1, dy=-0.55)
arrow(ax, (5.0, 1.7), (1.3, 1.7), RED, "", rad=-0.4, ls="--")
ax.text(3.15, 0.5, "BLOCKED: the model is told, and must go on without it", ha="center", fontsize=9, color=RED)
save(fig, "guardrail.png")
print("ok", [p.name for p in IMG.glob("*.png")])
