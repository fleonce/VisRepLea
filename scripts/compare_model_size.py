import matplotlib as mpl
import matplotlib.pyplot as plt
from with_argparse import with_argparse


mpl.rcParams["text.usetex"] = True


@with_argparse
def compare_model_size():
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    cmap = mpl.colormaps["tab20c"].colors

    ijepa = ax.bar(
        0.5,
        630762240,
        width=0.65,
        linewidth=0.5,
        edgecolor="black",
        color=cmap[6],
        label="abcdefg",
        hatch="//",
    )
    ax.bar_label(ijepa, [f"{630762240 / 1e6:.2f} M"], padding=4)
    clip = ax.bar(
        1.5,
        303179775,
        width=0.65,
        linewidth=0.5,
        edgecolor="black",
        color=cmap[2],
        label="abcdefg",
        hatch="//",
    )
    ax.bar_label(clip, [f"{303179775 / 1e6:.2f} M"], padding=4)

    ax.set_yscale("log")
    ax.set_ylim((1.5e8, 8e8))
    ax.set_xticks([0.5, 1.5], ["I-JEPA", "CLIP"])
    ax.set_ylabel(r"\# Model Parameters")
    ax.set_xlabel("Pretrained Image Models")
    ax.set_title(r"I-JEPA is 2X CLIP in terms of size")
    fig.savefig("compare_model_size.pdf")
    pass


compare_model_size()
