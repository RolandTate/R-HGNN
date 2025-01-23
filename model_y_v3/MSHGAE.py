import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm
from dgl.nn.pytorch import GATConv, SAGEConv, GraphConv

from model_y_v3.MSHGDecoder import MSHGDecoder
from model_y_v3.RelationGraphConv import RelationGraphConv, RelationAttentionConv
from model_y_v3.HeteroConv import HeteroGraphConv
from model_y_v3.RelationCrossing import RelationCrossing
from model_y_v3.RelationFusing import RelationFusing


class MSHGEncoderLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, n_heads: int = 8, dropout: float = 0.2, negative_slope: float = 0.2,
                 residual: bool = True, norm: bool = False):
        """

        :param graph: a heterogeneous graph
        :param input_dim: int, node input dimension
        :param hidden_dim: int, node hidden dimension
        :param relation_input_dim: int, relation input dimension
        :param relation_hidden_dim: int, relation hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param residual: boolean, residual connections or not
        :param norm: boolean, layer normalization or not
        """
        super(MSHGEncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.norm = norm

        # srcnode in relation transformation layers of each type
        self.relation_src_node_transformation_layers = nn.ModuleDict({
            etype: nn.Linear(input_dim, hidden_dim * n_heads, bias=False)
            for etype in graph.etypes
        })

        # relation transformation parameters of each type, used as attention queries
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, n_heads * 2 * hidden_dim))
            for etype in graph.etypes
        })

        # relation propagation layer of each relation
        self.relation_propagation_layer = nn.ModuleDict({
            etype: nn.Linear(relation_input_dim, n_heads * relation_hidden_dim)
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationAttentionConv(hidden_dim=hidden_dim, num_heads=n_heads,
                                         dropout=dropout, negative_slope=negative_slope)
            for etype in graph.etypes
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, h: dict, relation_embedding: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param relation_target_node_features: dict, {relation_type: target_node_features shape (N_nodes, input_dim)},
               each value in relation_target_node_features represents the representation of target node features
        :param relation_embedding: embedding for each relation, dict, {etype: feature}
        :return: output_features: dict, {relation_type: target_node_features}
        """
        input_src = h

        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in h:
                input_dst[(srctype, etypye, dsttype)] = h[(srctype, etypye, dsttype)][
                                                        :graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = h

        # output_features, dict {(srctype, etypye, dsttype): target_node_features}
        # h = {ntype: self.node_transformation_layers[ntype](h[ntype]) for ntype in graph.ntypes}
        output_features, dst_nodes_after_transformation = self.hetero_conv(graph, input_src, input_dst, relation_embedding,
                                           self.relation_src_node_transformation_layers,
                                           self.relation_transformation_weight)

        output_features_dict = {}
        # different relations crossing layer
        for srctype, etype, dsttype in output_features:
            output_features_dict[(srctype, etype, dsttype)] = output_features[(srctype, etype, dsttype)]

        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])

        # relation features after relation crossing layer, {(srctype, etype, dsttype): target_node_features}
        # relation embeddings after relation update, {etype: relation_embedding}
        return output_features_dict, relation_embedding_dict, dst_nodes_after_transformation


class MSHGEncoder(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, negative_slope: float = 0.2, residual: bool = True, norm: bool = False):
        """

        :param graph: a heterogeneous graph
        :param input_dim_dict: node input dimension dictionary
        :param hidden_dim: int, node hidden dimension
        :param relation_input_dim: int, relation input dimension
        :param relation_hidden_dim: int, relation hidden dimension
        :param num_layers: int, number of stacked layers
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param residual: boolean, residual connections or not
        :param norm: boolean, layer normalization or not
        """
        super(MSHGEncoder, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.norm = norm

        self.node_transformation_layers = nn.ModuleDict({
            ntype: nn.Linear(input_dim, hidden_dim * n_heads, bias=False)
            for ntype in graph.ntypes
        })

        # each layer takes in the heterogeneous graph as input
        self.relation_layer = MSHGEncoderLayer(graph, input_dim, hidden_dim, relation_input_dim, relation_hidden_dim, n_heads,
                         dropout, negative_slope, residual, norm)

        # transformation matrix for relation representation
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, relation_hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        self.residual_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(1)) for ntype in graph.ntypes
        })

        # different relations fusing module
        self.relation_fusing = RelationFusing(node_hidden_dim=hidden_dim,
                                              relation_hidden_dim=relation_hidden_dim,
                                              num_heads=n_heads,
                                              dropout=dropout, negative_slope=negative_slope)

        self.GConv_Layer = GraphConv(in_feats=input_dim, out_feats=hidden_dim * n_heads, bias=False)

        self.feature_fusion_layers = nn.ModuleDict({ntype: nn.Linear(hidden_dim * n_heads , hidden_dim * n_heads) for ntype in graph.ntypes})

        self.norms = nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim * n_heads) for ntype in graph.ntypes})

        self.drop = nn.Dropout(dropout)


        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, graph, h: dict, relation_embedding: dict = None):
        """

        :param blocks: list of sampled dgl.DGLHeteroGraph
        :param feats: Dict[str, tensor(N_i, d_in)]
        :param relation_embedding: embedding for each relation, dict, {etype: feature} or None
        :return:
        """
        feats_src = h

        if graph.is_block:
            feats_dst = {}
            for stype, etype, dtype in h:
                feats_dst[dtype] = h[(stype, etype, dtype)][
                                                        :graph.number_of_dst_nodes(dtype)]
            # feats_dst = {dtype: h[(stype, etype, dtype)][:graph.num_dst_nodes(dtype)] for stype, etype, dtype in h}
        else:
            feats_dst = h

        # 对每种节点类型应用节点特征变换层
        # h = {
        #     ntype: self.node_transformation_layers[ntype](h[ntype])
        #     for ntype in graph.ntypes
        # }

        for stype, reltype, dtype in h:
            h[(stype, reltype, dtype)] = self.node_transformation_layers[dtype](
                h[(stype, reltype, dtype)])

        # node level convolution
        # g = graph.local_var()
        # [print(g.nodes[ntype].data['feat'].shape) for ntype in g.ntypes]
        # for ntype in g.ntypes:
        #     g.nodes[ntype].data.update({'feat': h[ntype]})
        # [print(g.nodes[ntype].data['feat'].shape) for ntype in g.ntypes]
        # g = dgl.to_homogeneous(g, ndata=['feat'])
        # node_level_features = F.relu(self.GConv_Layer(g, g.ndata['feat']).view(-1, self.hidden_dim * self.n_heads))

        # relation convolution

        relation_target_node_features, relation_embedding, dst_nodes_after_transformation = self.relation_layer(graph, h,
                                                                      relation_embedding)

        relation_fusion_embedding_dict = {}
        s_id = 0
        e_id = 0
        # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
        for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
            relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                  for stype, etype, dtype in relation_target_node_features}

            etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
            # 得到每个终节点相关的基于关系的源节点聚合表示

            dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]


            # Tensor, shape (heads_num * hidden_dim)
            dst_node_relation_fusion_feature = self.relation_fusing(dst_node_features)

            e_id = e_id + graph.num_nodes(dsttype)
            # relation_fusion_embedding_dict[dsttype] = torch.cat([dst_node_relation_fusion_feature, node_level_features[s_id: e_id]], dim=-1)
            relation_fusion_embedding_dict[dsttype] = dst_node_relation_fusion_feature
            relation_fusion_embedding_dict[dsttype] = self.feature_fusion_layers[dsttype](relation_fusion_embedding_dict[dsttype])

            beta = F.sigmoid(self.residual_weight[dsttype])
            trans_out = self.drop(relation_fusion_embedding_dict[dsttype])
            # trans_out = relation_fusion_embedding_dict[dsttype]
            out = beta * trans_out + (1 - beta) * feats_dst[dsttype]
            # out = trans_out + feats_dst[dsttype]
            # relation_fusion_embedding_dict[dsttype] = self.norms[dsttype](out)
            relation_fusion_embedding_dict[dsttype] = out

            s_id = s_id + graph.num_nodes(dsttype)
            # 用于节点级编码y
            # graph.nodes[dsttype].data.update({'x': relation_fusion_embedding_dict[dsttype]})

        # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
        # relation_target_node_features, {(srctype, etype, dsttype): (dst_nodes, n_heads * hidden_dim)}
        return relation_fusion_embedding_dict, relation_embedding


class MSHGAE(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, negative_slope: float = 0.2, residual: bool = True, norm: bool = False):
        super(MSHGAE, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual
        self.norm = norm

        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # relation embedding dictionary
        self.relation_embedding = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, 1)) for etype in graph.etypes
        })

        self.encoder = nn.ModuleList()
        self.encoder.append(
            MSHGEncoder(graph=graph,
                  input_dim= hidden_dim * n_heads,
                  hidden_dim=hidden_dim, relation_input_dim=relation_input_dim,
                  relation_hidden_dim=relation_hidden_dim,
                  num_layers=1, n_heads=n_heads, dropout=dropout,
                  residual=residual))
        for _ in range(num_layers - 1):
            self.encoder.append(
                MSHGEncoder(graph=graph,
                  input_dim= hidden_dim * n_heads,
                  hidden_dim=hidden_dim, relation_input_dim=relation_hidden_dim * n_heads,
                  relation_hidden_dim=relation_hidden_dim,
                  num_layers=1, n_heads=n_heads, dropout=dropout,
                  residual=residual))

        self.decoder = nn.ModuleList()
        self.decoder.append(
            MSHGDecoder(graph=graph,
                        in_dim=hidden_dim * n_heads, out_dim=hidden_dim * n_heads,
                        num_heads=n_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.decoder.append(
                MSHGDecoder(graph=graph,
                            in_dim=hidden_dim * n_heads, out_dim=hidden_dim * n_heads,
                            num_heads=n_heads, dropout=dropout))



    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)

    def forward(self, blocks, inputs):
        # with graph.local_scope():
        if True:
            # initial projection
            for stype, reltype, dtype in inputs:
                inputs[(stype, reltype, dtype)] = self.projection_layer[dtype](
                    inputs[(stype, reltype, dtype)])

            # h = {ntype: F.gelu(self.projection_layer[ntype](inputs[ntype])) for ntype in blocks[0].ntypes}

            # [print(h[ntype].shape) for ntype in blocks[0].srctypes]
            # [print(blocks[0].nodes[ntype].data['feat'].shape) for ntype in blocks[0].dsttypes]
            # [print(blocks[1].nodes[ntype].data['feat'].shape) for ntype in blocks[0].dsttypes]

            # h = {ntype: self.projection_layer[ntype](graph.nodes[ntype].data['x']) for ntype in graph.ntypes}
            h = inputs
            transformed_h = inputs


            # each relation is associated with a specific type, if no semantic information is given,
            # then the one-hot representation of each relation is assign with trainable hidden representation
            relation_embedding = {etype: self.relation_embedding[etype].flatten() for etype in self.relation_embedding}

            for layer in self.encoder:
                for i, block in enumerate(blocks):
                    h, relation_embedding = layer(block, h, relation_embedding)

            for layer in self.decoder:
                for i, block in enumerate(blocks):
                    h = layer(block, h)

            # for layer in self.encoder:
            #     h, relation_embedding = layer(graph, h, relation_embedding)
            #
            # for layer in self.decoder:
            #     h = layer(graph, h)

        return h, transformed_h
