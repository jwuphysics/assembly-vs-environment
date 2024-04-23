import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, SAGEConv

class EdgeInteractionLayer(MessagePassing):
    """Graph interaction layer that combines node & edge features on edges.
    """
    def __init__(
        self, 
        n_in: int, 
        n_hidden: int, 
        n_latent: int, 
        aggr: list[str]=["sum", "max", "mean"], 
        act_fn: nn.Module=nn.SiLU
    ):
        super(EdgeInteractionLayer, self).__init__(aggr)

        self.mlp = nn.Sequential(
            nn.Linear(n_in, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.LayerNorm(n_hidden),
            act_fn(),
            nn.Linear(n_hidden, n_latent, bias=True),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(inputs)


class EdgeInteractionGNN(nn.Module):
    """Graph net over nodes and edges with multiple unshared layers, and sequential layers with residual connections.
    """
    def __init__(
        self, 
        n_layers: int, 
        node_features: int=2, 
        edge_features: int=6, 
        hidden_channels: int=64, 
        aggr: list[str]=["sum", "max", "mean"], 
        latent_channels: int=64, 
        n_out: int=1, 
        n_unshared_layers: int=4, 
        act_fn: nn.Module=nn.SiLU
    ):
        super(EdgeInteractionGNN, self).__init__()

        self.n_in = 2 * node_features + edge_features 
        self.n_out = n_out
        self.n_pool = (len(aggr) if isinstance(aggr, list) else 1) 
        
        layers = [
            nn.ModuleList([
                EdgeInteractionLayer(self.n_in, hidden_channels, latent_channels, aggr=aggr, act_fn=act_fn)
                for _ in range(n_unshared_layers)
            ])
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.ModuleList([
                    EdgeInteractionLayer(
                        self.n_pool * (2 * latent_channels * n_unshared_layers) + edge_features, 
                        hidden_channels, 
                        latent_channels, 
                        aggr=aggr, 
                        act_fn=act_fn
                    ) for _ in range(n_unshared_layers)
                ])
            ]
   
        self.layers = nn.ModuleList(layers)
        
        self.galaxy_environment_mlp = nn.Sequential(
            nn.Linear(self.n_pool * n_unshared_layers * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.galaxy_halo_mlp = nn.Sequential(
            nn.Linear(node_features, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, latent_channels, bias=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * latent_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.LayerNorm(hidden_channels),
            act_fn(),
            nn.Linear(hidden_channels, 2 * n_out, bias=True)
        )
    
    def forward(self, data: Data):
        
        # determine edges by getting neighbors within radius defined by `D_link`
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # update hidden state on edge (h, or sometimes e_ij in the text)
        h = torch.cat(
            [
                unshared_layer(data.x, edge_index=edge_index, edge_attr=edge_attr)
                for unshared_layer in self.layers[0]
            ], 
            axis=1
        )
        
        for layer in self.layers[1:]:
            # if multiple layers deep, also use a residual layer
            h += torch.cat(
                [
                    unshared_layer(h, edge_index=edge_index, edge_attr=edge_attr) 
                    for unshared_layer in layer
                ], 
                axis=1
            )
                        
        out =  torch.cat([self.galaxy_environment_mlp(h), self.galaxy_halo_mlp(data.x)], axis=1)
        return self.fc(out)

class SAGEGraphConvNet(torch.nn.Module):
    """A simple GNN built using SAGEConv layers.
    """
    def __init__(self, n_in=4, n_hidden=16, n_out=1, aggr="max"):
        super(SAGEGraphConvNet, self).__init__()
        self.conv1 = SAGEConv(n_in, n_hidden, aggr=aggr)
        self.conv2 = SAGEConv(n_hidden, n_hidden, aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(n_in + 2 * n_hidden, n_hidden, bias=True),
            nn.SiLU(),
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden, bias=True)
        )
        self.readout = nn.Linear(2 * n_hidden, 2 * n_out, bias=True)

    def forward(self, data):
        x0, edge_index = data.x, data.edge_index

        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(F.silu(x1), edge_index)
        out = self.mlp(torch.cat([F.silu(x2), F.silu(x1), x0], dim=-1))

        out = torch.cat([
            global_mean_pool(out, data.batch),
            global_max_pool(out, data.batch), 
        ], axis=1)
        
        return self.readout(out)

class MultiSAGENet(torch.nn.Module):
    """A multi-layer GNN built using SAGEConv layers.

    Makes full graph-level predictions (i.e. one for each merger tree). 

    TODO: implement edge features...
    """
    def __init__(
        self, 
        n_in=4, 
        n_hidden=16, 
        n_out=1, 
        n_layers=4, 
        aggr=["max", "sum", "mean"]
    ):
        super(MultiSAGENet, self).__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.aggr = aggr

        sage_convs = [SAGEConv(self.n_in, self.n_hidden, aggr=self.aggr)]                
        sage_convs += [SAGEConv(self.n_hidden, self.n_hidden, aggr=self.aggr) for _ in range(self.n_layers - 1)]
        self.convs = nn.ModuleList(sage_convs)

        self.mlp = nn.Sequential(
            nn.Linear(self.n_hidden, 4 * self.n_hidden, bias=True),
            nn.SiLU(),
            nn.LayerNorm(4 * self.n_hidden),
            nn.Linear(4 * self.n_hidden, self.n_hidden, bias=True)
        )
        self.readout = nn.Sequential(
            nn.Linear(3 * self.n_hidden, 4 * self.n_hidden, bias=True),
            nn.SiLU(),
            nn.LayerNorm(4 * self.n_hidden),
            nn.Linear(4 * self.n_hidden, 2 * self.n_out, bias=True)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.silu(x)

        x = self.mlp(x)
 
        out = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch), 
            global_add_pool(x, batch)
        ], axis=1)
              
        return self.readout(out)