import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """

        :param node_hidden_dim: int, node hidden feature size
        :param relation_hidden_dim: int,relation hidden feature size
        :param num_heads: int, number of heads in Multi-Head Attention
        :param dropout: float, dropout rate, defaults: 0.0
        :param negative_slope: float, negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)


    def forward(self, dst_node_features: list, raw_dst_node_features: list, dst_relation_embeddings: list,
                q_linear: nn.Linear, k_linears: list, v_linears: list,
                dst_relation_embedding_transformation_weight: list, residual_weight: nn.parameter):
        """
        :param dst_node_features: list, [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        :param dst_relation_embeddings: list, [each shape is (n_heads * relation_hidden_dim)]
        :param dst_node_feature_transformation_weight: list, [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        :param dst_relation_embedding_transformation_weight:  list, [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]
        :return: dst_node_relation_fusion_feature: Tensor of the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # dst_node_features.shape: torch.Size([4, 1280, 512]), reshape: torch.Size([4, 1280, 8, 64])
            # dst_relation_embeddings.shape: torch.Size([4, 64]), reshape: torch.Size([4, 8, 8])
            # dst_node_feature_transformation_weight.shape: torch.Size([4, 8, 64, 64]), reshape: torch.Size(
            #     [4, 8, 64, 64])
            # dst_relation_embedding_transformation_weight.shape: torch.Size([4, 8, 8, 64]), reshape: torch.Size(
            #     [4, 8, 8, 64])
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            raw_dst_node_features = torch.stack(raw_dst_node_features, dim=0)
            raw_dst_node_features = raw_dst_node_features.mean(dim=0)
            # q = [q_linear(raw_dst) for raw_dst, q_linear in zip(raw_dst_node_features, q_linears)]
            # q = [q_linear(raw_dst_node_features) for q_linear in q_linears]
            q = q_linear(raw_dst_node_features)
            k = [k_linear(dst_node_feature) for dst_node_feature, k_linear in zip(dst_node_features, k_linears)]
            v = [v_linear(dst_node_feature) for dst_node_feature, v_linear in zip(dst_node_features, v_linears)]

            # q = torch.stack(q, dim=0).view(len(q), -1, self.num_heads, self.node_hidden_dim)
            q = q.view(-1, self.num_heads, self.node_hidden_dim)
            k = torch.stack(k, dim=0).view(len(k), -1, self.num_heads, self.node_hidden_dim)
            v = torch.stack(v, dim=0).view(len(v), -1, self.num_heads, self.node_hidden_dim)

            # (num_dst_relations, n_heads, relation_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
                                                                                          self.num_heads,
                                                                                          self.relation_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
                                                                       dim=0).reshape(len(dst_node_features),
                                                                                      self.num_heads,
                                                                                      self.relation_hidden_dim,
                                                                                      self.node_hidden_dim)
            # shape (num_dst_relations, n_heads, hidden_dim)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
                                                   dst_relation_embedding_transformation_weight)
            k = (k * dst_relation_embeddings.unsqueeze(dim=1))
            t = torch.matmul(q, k.transpose(-1, -2)).sum(dim=-1, keepdim=True)
            # t = (q * k).sum(dim=1, keepdim=True)
            attn = t.squeeze(dim=-1) * 1 / math.sqrt(self.node_hidden_dim)
            attn = F.softmax(attn, dim=0)
            attn = attn.unsqueeze(dim=-1)
            dst_node_relation_fusion_feature = (v * attn).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1,
                                                                                        self.num_heads * self.node_hidden_dim)

            # dst_node_relation_fusion_feature.shape: torch.Size([1280, 512])
            # raw_dst_node_features.shape: torch.Size([4, 1280, 512])
            alpha = F.sigmoid(residual_weight)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature * alpha + raw_dst_node_features * (1 - alpha)
            # dst_node_relation_fusion_feature = dst_node_relation_fusion_feature + raw_dst_node_features
        return dst_node_relation_fusion_feature
