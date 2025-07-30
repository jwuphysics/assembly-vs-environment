import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, SAGEConv
from torch_geometric.utils import subgraph

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
    """A multi-layer GNN built using SAGEConv layers -- for REGULAR merger trees.
    """
    def __init__(
        self, 
        n_in=4, 
        n_hidden=16, 
        n_out=2,
        n_layers=4, 
        act_fn=nn.SiLU(),
        aggr=["max", "mean"]
    ):
        super(MultiSAGENet, self).__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.aggr = aggr
        self.act_fn = act_fn
        
        self.init_conv = SAGEConv(self.n_in, self.n_hidden, aggr=self.aggr)
        self.convs = nn.ModuleList()
        
        for _ in range(self.n_layers - 1):
            self.convs.append(SAGEConv(self.n_hidden, self.n_hidden, aggr=self.aggr))
            
        self.res_connection_adapter = nn.Linear(self.n_in, self.n_hidden)

        # activation funcs
        self.acts = nn.ModuleList()
        for _ in range(self.n_layers):
            self.acts.append(self.act_fn)
        
        # layernorms
        self.lns = nn.ModuleList() 
        for _ in range(self.n_layers):
            self.lns.append(nn.LayerNorm(self.n_hidden))

        # Node-level MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.n_hidden, 4 * self.n_hidden, bias=True),
            self.act_fn,
            nn.LayerNorm(4 * self.n_hidden),
            nn.Linear(4 * self.n_hidden, self.n_hidden, bias=True)
        )
        
        self.readout = nn.Sequential(
            nn.Linear(2 * self.n_hidden + 2 * self.n_in, 4 * self.n_hidden, bias=True),
            self.act_fn,
            nn.LayerNorm(4 * self.n_hidden),
            nn.Linear(4 * self.n_hidden, 2 * self.n_out, bias=True) 
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_res = self.res_connection_adapter(x)
        x = self.init_conv(x, edge_index)
        x = self.lns[0](x)
        x = x + x_res
        x = self.acts[0](x)

        for i in range(self.n_layers - 1):
            x_res = x  
            x = self.convs[i](x, edge_index)
            x = self.lns[i+1](x)
            x = x + x_res 
            x = self.acts[i+1](x)

        x = self.mlp(x)

        # batching to get root node for original features
        root_node_indices = data.ptr[:-1]
        original_root_features = data.x[root_node_indices]

        # get max of all original features
        x_max = global_max_pool(data.x, batch)

        # pooling + root node features
        out = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            # global_add_pool(x, batch),
            original_root_features,
            x_max
        ], dim=1)
                      
        return self.readout(out)

class ModernSAGEBlock(nn.Module):
    """
    A modern SAGE block with Pre-Normalization and two sub-layers:
    - SAGEConv for message passing (information mixing)
    - An MLP for node-wise feature processing (non-linearity)
    """
    def __init__(self, channels, aggr=["max", "mean"], act_fn=nn.ReLU(), dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(channels)
        self.conv = SAGEConv(channels, channels, aggr=aggr)
        
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            act_fn,
            nn.Linear(4 * channels, channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, h, edge_index):
        h_residual_1 = h
        h = self.norm1(h)
        h = self.conv(h, edge_index)
        h = h_residual_1 + h
        
        h_residual_2 = h
        h = self.norm2(h)
        h = self.mlp(h)
        h = h_residual_2 + h
        
        return h

class ModernSAGENet(torch.nn.Module):
    def __init__(
        self, 
        n_in=4, 
        n_hidden=16, 
        n_out=2,
        n_layers=4, 
        act_fn=nn.ReLU(),
        aggr=["max", "mean"]
    ):
        super().__init__()
        
        self.input_encoder = nn.Linear(n_in, n_hidden)
        
        self.gnn_blocks = nn.ModuleList(
            [ModernSAGEBlock(n_hidden, aggr, act_fn) for _ in range(n_layers)]
        )
        
        readout_in_dim = (2 * n_hidden) + (2 * n_in)
        self.readout = nn.Sequential(
            nn.Linear(readout_in_dim, 4 * n_hidden, bias=True),
            act_fn,
            nn.LayerNorm(4 * n_hidden),
            nn.Linear(4 * n_hidden, 2 * n_out, bias=True)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_encoder(x)
        
        for block in self.gnn_blocks:
            x = block(x, edge_index)
            
        root_node_indices = data.ptr[:-1]
        original_root_features = data.x[root_node_indices]
        x_max = global_max_pool(data.x, batch)

        out = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            original_root_features,
            x_max
        ], dim=1)
                      
        return self.readout(out)

######################
#   BONSAI + STUMP   #
###################### 
class SAGEEncoder(torch.nn.Module):
    """Simple SAGE GNN encoder."""
    def __init__(self, n_in=4, n_hidden=16, n_layers=4, aggr=["max", "mean"]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        
        self.init_conv = SAGEConv(n_in, n_hidden, aggr=aggr)
        self.lns.append(nn.LayerNorm(n_hidden))
        
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(n_hidden, n_hidden, aggr=aggr))
            self.lns.append(nn.LayerNorm(n_hidden))
            
        self.res_connection_adapter = nn.Linear(n_in, n_hidden)
        
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, 4 * n_hidden),
            nn.SiLU(),
            nn.LayerNorm(4 * n_hidden),
            nn.Linear(4 * n_hidden, n_hidden)
        )

    def forward(self, x, edge_index):
        x_res = self.res_connection_adapter(x)
        x = self.init_conv(x, edge_index)
        x = self.lns[0](x)
        x = x + x_res
        x = F.silu(x)

        for i in range(len(self.convs)):
            x_res = x  
            x = self.convs[i](x, edge_index)
            x = self.lns[i+1](x)
            x = x + x_res 
            x = F.silu(x)

        return self.mlp(x)

class BonsaiStumpSAGENet(torch.nn.Module):
    """
    A two-tower SAGE GNN to jointly process bonsai (trimmed) and stump (snapshots 90-99) graphs.
    """
    def __init__(self, n_in=12, n_hidden=32, n_out=2, n_layers=4, aggr=["max", "mean"]):
        super().__init__()
        
        self.bonsai_encoder = SAGEEncoder(n_in, n_hidden, n_layers, aggr)
        self.stump_encoder = SAGEEncoder(n_in, n_hidden, n_layers, aggr)
        
        readout_in_features = 4 * n_hidden + 2 * n_in
        
        self.readout = nn.Sequential(
            nn.Linear(readout_in_features, 4 * n_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(4 * n_hidden, 4 * n_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(4 * n_hidden, 2 * n_out, bias=True)
        )

    def forward(self, data):
        # bonsai encoder 
        x_bonsai, edge_index_bonsai, batch_bonsai = data['bonsai'].x, data['bonsai', 'to', 'bonsai'].edge_index, data['bonsai'].batch
        bonsai_node_feats = self.bonsai_encoder(x_bonsai, edge_index_bonsai)
        
        bonsai_pooled = torch.cat([
            global_mean_pool(bonsai_node_feats, batch_bonsai),
            global_max_pool(bonsai_node_feats, batch_bonsai),
            # global_add_pool(bonsai_node_feats, batch_bonsai),
        ], dim=1)

        # stump encoder
        x_stump, edge_index_stump, batch_stump = data['stump'].x, data['stump', 'to', 'stump'].edge_index, data['stump'].batch
        stump_node_feats = self.stump_encoder(x_stump, edge_index_stump)
        
        stump_pooled = torch.cat([
            global_mean_pool(stump_node_feats, batch_stump),
            global_max_pool(stump_node_feats, batch_stump),
            # global_add_pool(stump_node_feats, batch_stump),
        ], dim=1)

        # skip connection using pointer back to original root node's features
        bonsai_root_indices = data['bonsai'].ptr[:-1]
        bonsai_original_root_feats = x_bonsai[bonsai_root_indices]

        # also use max features from stump        
        stump_max_feats = global_max_pool(x_stump, batch_stump)
        
        # concat features
        combined_features = torch.cat([
            bonsai_pooled, 
            stump_pooled,
            bonsai_original_root_feats,
            stump_max_feats
        ], dim=1)
                      
        return self.readout(combined_features)

class ModernSAGEEncoder(torch.nn.Module):
    """
    A modern SAGE GNN encoder that stacks ModernSAGEBlocks.
    """
    def __init__(self, n_in=12, n_hidden=32, n_layers=4, aggr=["max", "mean"], act_fn=nn.ReLU()):
        super().__init__()
        
        self.input_encoder = nn.Linear(n_in, n_hidden)
        self.gnn_blocks = nn.ModuleList([ModernSAGEBlock(n_hidden, aggr, act_fn) for _ in range(n_layers)])
        
    def forward(self, x, edge_index):
        x = self.input_encoder(x)
        for block in self.gnn_blocks:
            x = block(x, edge_index)
        return x
        
class ModernBonsaiStumpSAGENet(torch.nn.Module):
    """
    A modernized two-tower SAGE GNN using ModernSAGEEncoder.
    """
    def __init__(self, n_in=12, n_hidden=32, n_out=2, n_layers=4, aggr=["max", "mean"], act_fn=nn.ReLU()):
        super().__init__()
        
        self.bonsai_encoder = ModernSAGEEncoder(n_in, n_hidden, n_layers, aggr, act_fn)
        self.stump_encoder = ModernSAGEEncoder(n_in, n_hidden, n_layers, aggr, act_fn)

        readout_in_features = (2 * n_hidden) + (2 * n_hidden) + n_in + n_in
        
        self.readout = nn.Sequential(
            nn.Linear(readout_in_features, 4 * n_hidden, bias=True),
            act_fn,
            nn.Linear(4 * n_hidden, 4 * n_hidden, bias=True),`
            act_fn,
            nn.Linear(4 * n_hidden, 2 * n_out, bias=True)
        )

    def forward(self, data):
        # pruned bonsai GNN
        x_bonsai, edge_index_bonsai, batch_bonsai = data['bonsai'].x, data['bonsai', 'to', 'bonsai'].edge_index, data['bonsai'].batch
        bonsai_node_feats = self.bonsai_encoder(x_bonsai, edge_index_bonsai)
        
        bonsai_pooled = torch.cat([
            global_mean_pool(bonsai_node_feats, batch_bonsai),
            global_max_pool(bonsai_node_feats, batch_bonsai),
        ], dim=1)

        # truncated stump GNN
        x_stump, edge_index_stump, batch_stump = data['stump'].x, data['stump', 'to', 'stump'].edge_index, data['stump'].batch
        stump_node_feats = self.stump_encoder(x_stump, edge_index_stump)
        
        stump_pooled = torch.cat([
            global_mean_pool(stump_node_feats, batch_stump),
            global_max_pool(stump_node_feats, batch_stump),
        ], dim=1)

        # get snapshot99 + max subhalo feats
        bonsai_root_indices = data['bonsai'].ptr[:-1]
        bonsai_original_root_feats = x_bonsai[bonsai_root_indices]
        stump_max_feats = global_max_pool(x_stump, batch_stump)
        
        combined_features = torch.cat([
            bonsai_pooled, 
            stump_pooled,
            bonsai_original_root_feats,
            stump_max_feats
        ], dim=1)
 
        return self.readout(combined_features)


    def forward(self, data):
        # Bonsai tower - uses the modern encoder
        x_bonsai, edge_index_bonsai, batch_bonsai = data['bonsai'].x, data['bonsai', 'to', 'bonsai'].edge_index, data['bonsai'].batch
        bonsai_node_feats = self.bonsai_encoder(x_bonsai, edge_index_bonsai)
        
        bonsai_pooled = torch.cat([
            global_mean_pool(bonsai_node_feats, batch_bonsai),
            global_max_pool(bonsai_node_feats, batch_bonsai),
        ], dim=1)

        # Stump tower - uses the modern encoder
        x_stump, edge_index_stump, batch_stump = data['stump'].x, data['stump', 'to', 'stump'].edge_index, data['stump'].batch
        stump_node_feats = self.stump_encoder(x_stump, edge_index_stump)
        
        stump_pooled = torch.cat([
            global_mean_pool(stump_node_feats, batch_stump),
            global_max_pool(stump_node_feats, batch_stump),
        ], dim=1)

        # Skip connection logic remains the same (it's domain-specific)
        bonsai_root_indices = data['bonsai'].ptr[:-1]
        bonsai_original_root_feats = x_bonsai[bonsai_root_indices]
        stump_max_feats = global_max_pool(x_stump, batch_stump)
        
        # Concatenate all features for the final prediction
        combined_features = torch.cat([
            bonsai_pooled, 
            stump_pooled,
            bonsai_original_root_feats,
            stump_max_feats
        ], dim=1)
                      
        return self.readout(combined_features)