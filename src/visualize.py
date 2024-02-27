import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Optional, Union

ROOT = Path(__file__).parent.parent.resolve()


def plot_merger_tree(
    tree: pd.DataFrame, 
    save_figname: Optional[Path]=None,
    arrows: bool=False,
    title: Optional[str]=None,
    cmap: matplotlib.colors.Colormap=plt.cm.viridis, 
    vmin: float=8, 
    vmax: float=11.5,
) -> None:
    """Plot 
    """
    plt.figure(figsize=(6, 6), dpi=300)

    edges = [(s, d) for s, d in zip(tree.subhalo_tree_id.values, tree.descendent_id.values) if d != -1]

    G = nx.DiGraph(edges)
    pos = graphviz_layout(G, prog="dot")
    m = tree.subhalo_loghalomass.values

    print(len(edges), m.shape)

    nx.draw(
        G, 
        pos=pos, 
        arrows=arrows, 
        node_color=m, 
        node_size=(m-7)**3, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = plt.colorbar(
        sm,
        ax=plt.gca(),
        label=r"$\log(M_{\rm halo}/M_\odot)$", 
        aspect=60, 
        shrink=0.8
    )

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if save_figname is not None:
        plt.savefig(f"{ROOT}/results/figures/{save_figname}.png")