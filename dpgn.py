import torch
import torch.nn as nn
import torch.nn.functional as F
from mca import SGA
from DAL import WSDAN

class PointSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(PointSimilarity, self).__init__()
        self.in_c = in_c  # 128
        self.base_c = base_c  # 128
        self.dropout = dropout  # 0.1
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),  # in:128 out:256
                       nn.BatchNorm2d(num_features=self.base_c*2),  # dim=256
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),  # in:256 out:128
                       nn.BatchNorm2d(num_features=self.base_c),  # dim=128
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]  # in:128 out:1
        self.point_sim_transform = nn.Sequential(*layer_list)  # 这个就是文中两层 conv2d-bn-relu 的相似度度量模块
        # self.relu = nn.ReLU()
        # self.dm = DiffusionMap(n_evecs=29, alpha=1.0, epsilon='bgh')

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """
        # conv = nn.Conv1d(vp_last_gen.size(1) - 1, vp_last_gen.size(1), kernel_size=1)
        # conv = conv.to(ep_last_gen.get_device())

        vp_i = vp_last_gen.unsqueeze(2)  # [25,10,1,128]
        vp_j = torch.transpose(vp_i, 1, 2)  # 25,1,10,128 
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2   # [25,10,10,128]
            # vp_similarity = 1 - F.cosine_similarity(vp_j, vp_j, 3)
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))   # 边就是相似度来计算的

        # normalizatin
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(ep_last_gen.get_device())
        ep_last_gen *= diagonal_mask
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)  # [25,10,10]

        return ep_ij, node_similarity_l2


# class L_PointSimilarity(nn.Module):
#     def __init__(self, in_c, base_c, dropout=0.0):
#         super(L_PointSimilarity, self).__init__()
#         self.in_c = in_c  # 128
#         self.base_c = base_c  # 128
#         self.dropout = dropout  # 0.1
#         layer_list = []
#
#         layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),  # in:128 out:256
#                        nn.BatchNorm2d(num_features=self.base_c*2),  # dim=256
#                        nn.LeakyReLU()]
#
#         if self.dropout > 0:
#             layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1
#
#         layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),  # in:256 out:128
#                        nn.BatchNorm2d(num_features=self.base_c),  # dim=128
#                        nn.LeakyReLU()]
#
#         if self.dropout > 0:
#             layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1
#
#         layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]  # in:128 out:1
#         self.point_sim_transform = nn.Sequential(*layer_list)  # 这个就是文中两层 conv2d-bn-relu 的相似度度量模块
#         # self.relu = nn.ReLU()
#         # self.dm = DiffusionMap(n_evecs=29, alpha=1.0, epsilon='bgh')
#
#     def forward(self, vp_last_gen, ep_last_gen, distance_metric):
#         """
#         Forward method of Point Similarity
#         :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
#         :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
#         :param distance_metric: metric for distance
#         :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
#                  l2 version of node similarities
#         """
#         # conv = nn.Conv1d(vp_last_gen.size(1) - 1, vp_last_gen.size(1), kernel_size=1)
#         # conv = conv.to(ep_last_gen.get_device())
#
#         vp_i = vp_last_gen.unsqueeze(2)  # [25,10,1,128]
#         vp_j = torch.transpose(vp_i, 1, 2)  # 25,1,10,128
#         if distance_metric == 'l2':
#             vp_similarity = (vp_i - vp_j)**2   # [25,10,10,128]
#             # vp_similarity = 1 - F.cosine_similarity(vp_j, vp_j, 3)
#         elif distance_metric == 'l1':
#             vp_similarity = torch.abs(vp_i - vp_j)
#         trans_similarity = torch.transpose(vp_similarity, 1, 3)
#         ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))   # 边就是相似度来计算的
#
#         # noormalizatin
#         diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(ep_last_gen.get_device())
#         ep_last_gen *= diagonal_mask
#         ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
#         ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
#         diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
#         ep_ij += (diagonal_reverse_mask + 1e-6)
#         ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
#         node_similarity_l2 = -torch.sum(vp_similarity, 3)  # [25,10,10]
#
#         # dm = DiffusionMap(n_evecs=vp_last_gen.size(1) - 1, alpha=1.0, epsilon='bgh')
#
#         # 获取每个batch的点图节点特征
#         # batch_tensors = torch.unbind(vp_last_gen, dim=0)
#         # 对每个batch的图进行diffmap
#         # for i in range(len(batch_tensors)):
#         #     diffmap = None
#         #     has_nan = True
#         #     try:
#         #         diffmap = dm.fit_transform(batch_tensors[i]).to(vp_last_gen.get_device())
#         #         diffmap = diffmap.transpose(0, 1).unsqueeze(0)
#         #         diffmap = conv(diffmap).transpose(2, 1)
#         #         has_nan = torch.isnan(diffmap).any()
#         #
#         #     except Exception as e:
#         #         pass
#         #
#         #     if has_nan:
#         #         if i == 0:
#         #             diff = ep_ij[i].unsqueeze(0)
#         #         else:
#         #             diff = torch.cat([diff, ep_ij[i].unsqueeze(0)], dim=0)
#         #     else:
#         #         if i == 0:
#         #             diff = diffmap
#         #         else:
#         #             diff = torch.cat([diff, diffmap], dim=0)
#
#         return ep_ij, node_similarity_l2


class Initial(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(Initial, self).__init__()
        self.in_c = in_c  # 128
        self.base_c = base_c  # 128
        self.dropout = dropout  # 0.1
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),  # in:128 out:256
                       nn.BatchNorm2d(num_features=self.base_c*2),  # dim=256
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False),  # in:256 out:128
                       nn.BatchNorm2d(num_features=self.base_c),  # dim=128
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]  # in:128 out:1
        self.point_sim_transform = nn.Sequential(*layer_list)  # 这个就是文中两层 conv2d-bn-relu 的相似度度量模块

    def forward(self, vp_last_gen, ep_last_gen, distance_metric):
        """
        Forward method of Point Similarity
        :param vp_last_gen: last generation's node feature of point graph, Vp_(l-1)
        :param ep_last_gen: last generation's edge feature of point graph, Ep_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l) (for Point Loss)
                 l2 version of node similarities
        """

        vp_i = vp_last_gen.unsqueeze(2)  # [25,10,1,128]
        vp_j = torch.transpose(vp_i, 1, 2)  # 25,1,10,128
        if distance_metric == 'l2':
            vp_similarity = (vp_i - vp_j)**2   # [25,10,10,128]
            # vp_similarity = 1 - F.cosine_similarity(vp_j, vp_j, 3)
        elif distance_metric == 'l1':
            vp_similarity = torch.abs(vp_i - vp_j)
        trans_similarity = torch.transpose(vp_similarity, 1, 3)
        ep_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))   # 边就是相似度来计算的

        # noormalizatin
        diagonal_mask = 1.0 - torch.eye(vp_last_gen.size(1)).unsqueeze(0).repeat(vp_last_gen.size(0), 1, 1).to(ep_last_gen.get_device())
        ep_last_gen *= diagonal_mask
        ep_last_gen_sum = torch.sum(ep_last_gen, -1, True)
        ep_ij = F.normalize(ep_ij.squeeze(1) * ep_last_gen, p=1, dim=-1) * ep_last_gen_sum
        diagonal_reverse_mask = torch.eye(vp_last_gen.size(1)).unsqueeze(0).to(ep_last_gen.get_device())
        ep_ij += (diagonal_reverse_mask + 1e-6)
        ep_ij /= torch.sum(ep_ij, dim=2).unsqueeze(-1)
        node_similarity_l2 = -torch.sum(vp_similarity, 3)  # [25,10,10]

        return ep_ij, node_similarity_l2


class P2DAgg(nn.Module):
    def __init__(self, in_c, out_c):
        super(P2DAgg, self).__init__()
        # add the fc layer
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True), #in:10 out:5
                                             nn.LeakyReLU()])
        self.out_c = out_c  # 5
        # self.linear = nn.Linear(in_features=in_c*2, out_features=in_c)

    def forward(self, point_edge, distribution_node):
        # point_edge:[10,10,10]
        # 更新分布图的节点特征
        meta_batch = point_edge.size(0)
        num_sample = point_edge.size(1)
        # point_edge = self.linear(torch.cat([point_edge, lang_edge], dim=-1))

        distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
        distribution_node = distribution_node.view(meta_batch*num_sample, -1)
        distribution_node = self.p2d_transform(distribution_node)   # 线性层+ReLu层做转换
        distribution_node = distribution_node.view(meta_batch, num_sample, -1)
        return distribution_node


# class L_P2DAgg(nn.Module):
#     def __init__(self, in_c, out_c):
#         super(L_P2DAgg, self).__init__()
#         # add the fc layer
#         self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True), #in:10 out:5
#                                              nn.LeakyReLU()])
#         self.out_c = out_c  # 5
#         # self.linear = nn.Linear(in_features=in_c*2, out_features=in_c)
#
#     def forward(self, point_edge, distribution_node):
#         # point_edge:[10,10,10]
#         # 更新分布图的节点特征
#         meta_batch = point_edge.size(0)
#         num_sample = point_edge.size(1)
#         # point_edge = self.linear(torch.cat([point_edge, lang_edge], dim=-1))
#
#         distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)
#         distribution_node = distribution_node.view(meta_batch*num_sample, -1)
#         distribution_node = self.p2d_transform(distribution_node)   # 线性层+ReLu层做转换
#         distribution_node = distribution_node.view(meta_batch, num_sample, -1)
#         return distribution_node


# 分布相似度就是分布图的边
class DistributionSimilarity(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        """
        Distribution Similarity (see paper 3.2.2) Vd_(l) -> Ed_(l)
        :param in_c: number of input channel
        :param base_c: number of base channel
        :param device: the gpu device stores tensors
        :param dropout: dropout rate
        """
        super(DistributionSimilarity, self).__init__()
        self.in_c = in_c  # 5
        self.base_c = base_c  # 5
        self.dropout = dropout  # 0.1
        layer_list = []

        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),  # in:5 out:10
                       nn.BatchNorm2d(num_features=self.base_c * 2),  #dim:10
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),  # in:10 out:5
                       nn.BatchNorm2d(num_features=self.base_c),  # 5
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]  # in:5 out:1
        self.point_sim_transform = nn.Sequential(*layer_list)  # 更新 distribution graph 的 边相似度信息

    def forward(self, vd_curr_gen, ed_last_gen, distance_metric):
        """
        Forward method of Distribution Similarity
        :param vd_curr_gen: current generation's node feature of distribution graph, Vd_(l)
        :param ed_last_gen: last generation's edge feature of distribution graph, Ed_(l-1)
        :param distance_metric: metric for distance
        :return: edge feature of point graph in current generation Ep_(l)
        """
        vd_i = vd_curr_gen.unsqueeze(2)
        vd_j = torch.transpose(vd_i, 1, 2)
        if distance_metric == 'l2':
            vd_similarity = (vd_i - vd_j)**2
        elif distance_metric == 'l1':
            vd_similarity = torch.abs(vd_i - vd_j)
        trans_similarity = torch.transpose(vd_similarity, 1, 3)
        ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))

        # normalization
        diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(ed_last_gen.get_device())
        ed_last_gen *= diagonal_mask
        ed_last_gen_sum = torch.sum(ed_last_gen, -1, True)
        ed_ij = F.normalize(ed_ij.squeeze(1) * ed_last_gen, p=1, dim=-1) * ed_last_gen_sum
        diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(ed_last_gen.get_device())
        ed_ij += (diagonal_reverse_mask + 1e-6)
        ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)

        return ed_ij


# class L_DistributionSimilarity(nn.Module):
#     def __init__(self, in_c, base_c, dropout=0.0):
#         super(L_DistributionSimilarity, self).__init__()
#         self.in_c = in_c  # 5
#         self.base_c = base_c  # 5
#         self.dropout = dropout  # 0.1
#         layer_list = []
#
#         layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),  # in:5 out:10
#                        nn.BatchNorm2d(num_features=self.base_c * 2),  #dim:10
#                        nn.LeakyReLU()]
#
#         if self.dropout > 0:
#             layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1
#
#         layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),  # in:10 out:5
#                        nn.BatchNorm2d(num_features=self.base_c),  # 5
#                        nn.LeakyReLU()]
#
#         if self.dropout > 0:
#             layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1
#
#         layer_list += [nn.Conv2d(in_channels=self.base_c, out_channels=1, kernel_size=1)]  # in:5 out:1
#         self.point_sim_transform = nn.Sequential(*layer_list)  # 更新 distribution graph 的 边相似度信息
#
#     def forward(self, vd_curr_gen, ed_last_gen, distance_metric):
#         vd_i = vd_curr_gen.unsqueeze(2)
#         vd_j = torch.transpose(vd_i, 1, 2)
#         if distance_metric == 'l2':
#             vd_similarity = (vd_i - vd_j)**2
#         elif distance_metric == 'l1':
#             vd_similarity = torch.abs(vd_i - vd_j)
#         trans_similarity = torch.transpose(vd_similarity, 1, 3)
#         ed_ij = torch.sigmoid(self.point_sim_transform(trans_similarity))
#
#         # normalization
#         diagonal_mask = 1.0 - torch.eye(vd_curr_gen.size(1)).unsqueeze(0).repeat(vd_curr_gen.size(0), 1, 1).to(ed_last_gen.get_device())
#         ed_last_gen *= diagonal_mask
#         ed_last_gen_sum = torch.sum(ed_last_gen, -1, True)
#         ed_ij = F.normalize(ed_ij.squeeze(1) * ed_last_gen, p=1, dim=-1) * ed_last_gen_sum
#         diagonal_reverse_mask = torch.eye(vd_curr_gen.size(1)).unsqueeze(0).to(ed_last_gen.get_device())
#         ed_ij += (diagonal_reverse_mask + 1e-6)
#         ed_ij /= torch.sum(ed_ij, dim=2).unsqueeze(-1)
#
#         return ed_ij


class D2PAgg(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(D2PAgg, self).__init__()
        self.in_c = in_c  # 256
        self.base_c = base_c  # 128
        self.dropout = dropout  # 0.1
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c*2, kernel_size=1, bias=False),  # in:256 out:256
                       nn.BatchNorm2d(num_features=self.base_c*2), # dim:256
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c*2, out_channels=self.base_c, kernel_size=1, bias=False), # in:256 out:128
                       nn.BatchNorm2d(num_features=self.base_c), # dim:128
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]  # 0.1

        self.point_node_transform = nn.Sequential(*layer_list)  # 这个就是将前后两代 point graph 点特征 进行融合降维的模块
        self.linear1 = nn.Linear(base_c*2, base_c)
        self.linear2 = nn.Linear(300, base_c)

    def forward(self, distribution_edge, point_node):
        # 更新点图的节点特征
        # get size
        meta_batch = point_node.size(0)
        num_sample = point_node.size(1)

        # get eye matrix (batch_size x node_size x node_size)
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())

        # set diagonal as zero and normalize
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)

        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat, point_node)    #[10,10,128]

        node_feat = torch.cat([point_node, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))  #新旧点图节点做融合
        node_feat = node_feat.transpose(1, 2).squeeze(-1)
        # que_data = self.linear2(que_data)
        # node_feat = self.linear1(torch.cat([node_feat, que_data], dim=-1))

        return node_feat


class SemPointUpdate(nn.Module):
    def __init__(self, in_c, base_c, dropout=0.0):
        super(SemPointUpdate, self).__init__()
        self.in_c = in_c
        self.base_c = base_c
        self.dropout = dropout

        # 两个Conv-BN-ReLU的卷积块构成
        layer_list = []
        layer_list += [nn.Conv2d(in_channels=self.in_c, out_channels=self.base_c * 2, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c * 2),
                       nn.LeakyReLU()]

        layer_list += [nn.Conv2d(in_channels=self.base_c * 2, out_channels=self.base_c, kernel_size=1, bias=False),
                       nn.BatchNorm2d(num_features=self.base_c),
                       nn.LeakyReLU()]

        if self.dropout > 0:
            layer_list += [nn.Dropout2d(p=self.dropout)]

        self.point_node_transform = nn.Sequential(*layer_list)

    def forward(self, distribution_edge, instance_edge, point_node):
        meta_batch = point_node.size(0)  # batch size => 25
        num_sample = point_node.size(1)  # support + query => 10/30

        # 获得一个 1-单位矩阵 (batch_size x node_size x node_size)  [25,10,10]
        diag_mask = 1.0 - torch.eye(num_sample).unsqueeze(0).repeat(meta_batch, 1, 1).to(distribution_edge.get_device())

        # 将边的对角线全部设置为0 并且进行 normalization
        edge_feat = F.normalize(distribution_edge * diag_mask, p=1, dim=-1)  # [25,10,10]

        # distribution graph 的边 * point graph 的点 => attention
        aggr_feat = torch.bmm(edge_feat, point_node)

        # 将 重新计算后的信息 与 上一次迭代的点信息 进行 级联
        edge_instance = F.normalize(instance_edge * diag_mask, p=1, dim=-1)  # [25,10,10]
        aggr_instance = torch.bmm(edge_instance, point_node)
        node_feat = torch.cat([aggr_feat, aggr_instance], -1).transpose(1, 2)  # [25,128*2,10]

        # 放入 D2P 网络
        node_feat = self.point_node_transform(node_feat.unsqueeze(-1))  # [25,128,10,1]
        node_feat = node_feat.transpose(1, 2).squeeze(-1)  # [25,10,128]

        return node_feat


class LinearTransformation(nn.Module):          #P2DAgg
    def __init__(self, in_c, out_c):
        super(LinearTransformation, self).__init__()

        # FC+RELU
        self.p2d_transform = nn.Sequential(*[nn.Linear(in_features=in_c, out_features=out_c, bias=True),
                                             nn.LeakyReLU()])
        self.out_c = out_c

    def forward(self, point_edge, distribution_node):
        meta_batch = point_edge.size(0)  # batch size => 25
        num_sample = point_edge.size(1)  # support + query => 10

        # 先将 边集合 与 上一次迭代的点信息 级联起来
        distribution_node = torch.cat([point_edge[:, :, :self.out_c], distribution_node], dim=2)  # [25,10,10]
        distribution_node = distribution_node.view(meta_batch*num_sample, -1)  # [250,15]

        # 放入P2D网络
        distribution_node = self.p2d_transform(distribution_node)  # [250,5]
        distribution_node = distribution_node.view(meta_batch, num_sample, -1)  # [25,10,5]

        return distribution_node


class DPGN(nn.Module):
    def __init__(self, num_generations, dropout, num_way, num_shot, num_support_sample, num_sample, loss_indicator, point_metric, distribution_metric):
        super(DPGN, self).__init__()
        self.generation = num_generations  # 6
        self.dropout = dropout  # 0.1
        self.num_support_sample = num_support_sample  # 5
        self.num_sample = num_sample  # 5 + 5
        self.loss_indicator = loss_indicator  # [1,1,0]
        self.point_metric = point_metric  # 'l2'
        self.distribution_metric = distribution_metric  # ''l2
        self.num_way = num_way                           # 5
        self.num_shot = num_shot                         # 1/5
        self.c_att = WSDAN()

        self.linear1 = nn.Linear(2048, 512)     #batch, 50, 2048, 512
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(1024, 128)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(300*2, 300)
        # self.linear6 = nn.Linear(1024, 512)
        self.norm = nn.BatchNorm1d(512)
        self.cos = nn.CosineSimilarity(dim=-1)

        # node & edge update module can be formulated by yourselves
        P_Sim = PointSimilarity(128, 128, dropout=self.dropout)
        # INIT = Initial(128, 128, dropout=self.dropout)
        self.add_module('initial_edge', P_Sim)  # 添加当前模块到网络中
        # self.add_module('l_initial_edge', INIT)
        for l in range(self.generation): # self.generation => 6
            D2P = D2PAgg(128*2, 128, dropout=self.dropout if l < self.generation-1 else 0.0)  # 最后一层dropout=0.0
            P2D = P2DAgg(num_sample, num_support_sample)  # num_support_sample:5

            P_Sim = PointSimilarity(128, 128, dropout=self.dropout if l < self.generation-1 else 0.0)
            D_Sim = DistributionSimilarity(num_support_sample,  # num_support_sample:5
                                           num_support_sample,
                                            dropout=self.dropout if l < self.generation-1 else 0.0)

            # L_P2D = L_P2DAgg(num_sample, num_support_sample)  # num_support_sample:5
            # L_D2P = L_D2PAgg(300*2, 300, dropout=self.dropout if l < self.generation-1 else 0.0)
            # L_P_Sim = L_PointSimilarity(300, 300, dropout=self.dropout if l < self.generation-1 else 0.0)
            # L_D_Sim = L_DistributionSimilarity(num_support_sample,  # num_support_sample:5
            #                                    num_support_sample,
            #                                  dropout=self.dropout if l < self.generation-1 else 0.0)
            #
            # yuyi point graph
            sem_point_node_update = SemPointUpdate(300*2, 300, dropout=self.dropout if l < self.generation - 1 else 0.0)
            sem_point_edge_update = PointSimilarity(300, 300, dropout=self.dropout if l < self.generation - 1 else 0.0)

            # yuyi distribution graph
            sem_distribution_edge_update = DistributionSimilarity(num_support_sample, num_support_sample, dropout=self.dropout if l < self.generation-1 else 0.0)
            sem_distribution_node_update = LinearTransformation(num_sample, num_support_sample)

            self.add_module('point2distribution_generation_{}'.format(l), P2D)
            self.add_module('distribution2point_generation_{}'.format(l), D2P)
            self.add_module('point_sim_generation_{}'.format(l), P_Sim)
            self.add_module('distribution_sim_generation_{}'.format(l), D_Sim)
            # self.add_module('l_distribution_sim_generation_{}'.format(l), L_D_Sim)

            # self.add_module('l_distribution2point_generation_{}'.format(l), L_D2P)
            # self.add_module('l_point2distribution_generation_{}'.format(l), L_P2D)
            # self.add_module('l_point_sim_generation_{}'.format(l), L_P_Sim)

            self.add_module('sem_point_node_update{}'.format(l), sem_point_node_update)
            self.add_module('sem_point_edge_update{}'.format(l), sem_point_edge_update)
            self.add_module('sem_distribution_node_update{}'.format(l), sem_distribution_node_update)
            self.add_module('sem_distribution_edge_update{}'.format(l), sem_distribution_edge_update)

            self.add_module('edge_update{}'.format(l), P_Sim)

    def init_edge(self, nodes):
        edge_init = []
        for raw in nodes:
            sim = []
            for point in raw:
                sim.append(F.sigmoid(self.cos(point, raw)))
            sim = torch.stack(sim)
            edge_init.append(sim)
        edge_init = torch.stack(edge_init)
        edge_num = edge_init.shape[-1]

        for j in range(edge_num):
            edge_init[:, j, j] = 1.0

        return edge_init

    def get_proto_node_with_query(self, node_feat, edge_feat):
        num_supports = self.num_support_sample  # 5
        n_batch = node_feat.shape[0]   # 25/10

        # 查询的相似度，从边里提取出和支持样本相似度最大的索引号（这里输入的边是子图相似度的边）
        query_similarity = edge_feat[:, num_supports:, :num_supports]  # [25,5,5]/[25,25,5] [batch,support,query]
        query_node = node_feat[:, num_supports:, :]  # [25,5,128]
        index = torch.argmax(query_similarity, dim=-2)  # [25,5]/[25,25] 和support相似度最高的节点的index

        protos = []  # prototype
        for batch_index in range(0, n_batch):
            num = 0
            if self.num_way == 5:
                task_proto = [[], [], [], [], []]
            else:
                task_proto = [[], [], [], [], [], [], [], [], [], []]

            for query_index in index[batch_index]:
                current_index = num // self.num_shot
                task_proto[current_index].append(query_node[batch_index][query_index])
                num += 1
                if num % self.num_shot == 0:
                    task_proto[current_index] = torch.stack(task_proto[current_index])
            task_proto = torch.stack(task_proto)
            protos.append(task_proto)

        protos = torch.stack(protos)  # [25,5,1,128]/[25,5,5,128]
        support_node = node_feat[:, :num_supports, :]  # [25,5,128]/[25,25,128]
        support_node = support_node.view(n_batch, self.num_way, self.num_shot, -1)  # [25,5,1,128]/[25,5,5,128]
        protos = torch.cat((protos, support_node), dim=-2)  # [25,5,2,128]/[25,5,10,128]
        protos = protos.mean(dim=-2)  # [batch,num_way,128]

        return protos   #和support节点相似度最高的N-way个query节点拉出来并和support融合作为proto

    def get_proto_distribution(self, protos, node_feat):
        num_samples = self.num_sample
        if node_feat.shape[2] == 512:
            node_feat = self.linear4(node_feat)

        proto_distribution = None  # [25,10,5]   node-proto distribution
        for sample_index in range(0, num_samples):  #对所有sample都和proto做一次相似度
            current_nodes = node_feat[:, sample_index, :]  # [25,128]
            current_nodes = current_nodes.unsqueeze(1)  # [25,1,128]
            sim = self.cos(current_nodes, protos)  # [25,5]
            sim = sim.unsqueeze(1)  # [25,1,5]
            if proto_distribution is None:
                proto_distribution = sim
            else:
                proto_distribution = torch.cat((proto_distribution, sim), dim=-2)

        return proto_distribution  # [25,10,5]

    # def get_yuyi_proto_distribution(self, protos, cls):
    #     # cls = cls.sum(-2)
    #
    #     proto_distribution = None  # [25,10,5]   node-proto distribution
    #     for sample_index in range(0, self.num_sample):  #对所有sample都和proto做一次相似度
    #         current_nodes = cls[:, sample_index, :]  # [25,128]
    #         current_nodes = current_nodes.unsqueeze(1)  # [25,1,128]
    #         sim = self.cos(current_nodes, protos)  # [25,5]
    #         sim = sim.unsqueeze(1)  # [25,1,5]
    #         if proto_distribution is None:
    #             proto_distribution = sim
    #         else:
    #             proto_distribution = torch.cat((proto_distribution, sim), dim=1)
    #
    #     # proto_distribution = proto_distribution.sum(-1).squeeze(-1)
    #
    #     return proto_distribution  # [batch,sample, 10,50] cls的50个object对每一个sample的que_data的相似度

    def bridge_cls_que(self, que_data, cls):
        sample = self.num_sample

        sim = None
        for sample_index in range(sample):
            current_que = que_data[:, sample_index, :, :]
            que = torch.sum(current_que, dim=-2)
            current_cls = cls[:, sample_index, :, :]
        #     q_index = current_que.shape[1]
            c_index = current_cls.shape[1]
            c_sim = None
            for c in range(c_index):
                cos = self.cos(current_cls[:, c, :], que)
                if c_sim is None:
                    c_sim = cos.unsqueeze(1)
                else:
                    c_sim = torch.cat((cos.unsqueeze(1), c_sim), dim=1)

            if sim is None:
                sim = c_sim.unsqueeze(1)
            else:
                sim = torch.cat((c_sim.unsqueeze(1), sim), dim=1)

        value, index = torch.topk(sim, 15, -1)

        return index

    def bridge_cls_img(self, index, vinvl, cls):
        batch_size, sample_size, _, feature_dim = vinvl.shape
        cls_dim = cls.shape[-1]
        index_expanded = index.unsqueeze(-1).expand(-1, -1, -1, feature_dim)
        index_cls = index.unsqueeze(-1).expand(-1, -1, -1, cls_dim)
        temp = vinvl.gather(2, index_expanded)
        temp_cls = cls.gather(2,index_cls)

        selected_vinvl = temp.sum(2).view(batch_size*sample_size, -1)
        selected_vinvl = self.norm(self.linear1(selected_vinvl)).view(batch_size, sample_size, -1)

        select_cls = temp_cls.sum(2)

        return selected_vinvl, select_cls

    def forward(self, middle_node, point_node, que_data, distribution_node, vinvl_feat, cls, que_embedding, cls_embedding):

        # print(vinvl_feat)       # vinvl_feat:[batch, sample, 50, 2048]
        # print(cls)      # cls:[batch, sample, 50, 300]过了LSTM和SA后的特征
        # print(que_embedding.size()) # [10, 10, 15, 300]
        # print(cls_embedding.size()) # [10, 10, 50, 300]
        batch = vinvl_feat.size(0)
        sample = vinvl_feat.size(1)

        index = self.bridge_cls_que(que_embedding, cls_embedding)
        selected_vinvl, selected_cls = self.bridge_cls_img(index, vinvl_feat, cls)

        sem_point_node = torch.cat([que_data, selected_cls], -1)
        sem_point_node = self.linear5(sem_point_node)

        point_similarities = []
        distribution_similarities = []
        node_similarities_l2 = []
        sem_edge_similarities = []                 # yuyi graph edge loss

        # lang_edge = self.init_edge(que_data)
        # point_edge = self.init_edge(middle_node)
        sem_point_edge = self.init_edge(que_data)

        vi_distribution_node = distribution_node.detach()
        sem_distribution_node = distribution_node.detach()

        clip_node = self.linear2(point_node).view(-1, 512)
        clip_node = self.norm(clip_node).view(batch, sample, 512)
        # print(id(vi_distribution_node))
        # print(id(sem_distribution_node))

        for l in range(self.generation):

            # 边更新模块
            point_edge, node_similarity_l2 = self._modules['point_sim_generation_{}'.format(l)](point_node, sem_point_edge, self.point_metric)

            # lang_edge, _ = self._modules['l_point_sim_generation_{}'.format(l)](que_data, lang_edge, self.point_metric)
            sem_point_edge, _ = self._modules['sem_point_edge_update{}'.format(l)](sem_point_node, point_edge, self.point_metric)

            # vinvl和clip的融合模块，用来增强特征
            _, enhanced_node = self.c_att(selected_vinvl, clip_node, None, None)
            enhanced_node = self.linear3(enhanced_node)
            if l == 0:
                enhanced_edge = self.init_edge(enhanced_node)
            enhanced_edge, _ = self._modules['edge_update{}'.format(l)](enhanced_node, enhanced_edge, self.point_metric)

            # 增强后的节点和surport做相似度运算得到分布节点
            v_local_proto_node = self.get_proto_node_with_query(point_node, enhanced_edge)  # [batch, num_way, 128]
            v_local_proto_edge = self.get_proto_distribution(v_local_proto_node, point_node)  # [batch, num_sample, 5]

            # l_local_proto_node = self.get_proto_node_with_query(que_data, sem_point_edge)
            # l_local_proto_edge = self.get_proto_distribution(l_local_proto_node, sem_point_node)

            #vqa_clip
            l_local_proto_edge = self.get_proto_distribution(que_data[:,:self.num_support_sample:self.num_shot,:], sem_point_node)

            vi_distribution_node = self._modules['point2distribution_generation_{}'.format(l)](v_local_proto_edge, vi_distribution_node)    #vi_distribution_node：[batch, num_sample, num_support]
            sem_distribution_node = self._modules['sem_distribution_node_update{}'.format(l)](l_local_proto_edge, sem_distribution_node)

            if l == 0:
                vi_distribution_edge = self.init_edge(vi_distribution_node)
                sem_distribution_edge = self.init_edge(sem_distribution_node)
            vi_distribution_edge = self._modules['distribution_sim_generation_{}'.format(l)](vi_distribution_node, vi_distribution_edge, self.distribution_metric)
            sem_distribution_edge = self._modules['sem_distribution_edge_update{}'.format(l)](sem_distribution_node, sem_distribution_edge, self.distribution_metric)

            # (9) 更新 yuyi point node
            sem_point_node = self._modules['sem_point_node_update{}'.format(l)](sem_distribution_edge, sem_point_edge, sem_point_node)  # [25,10,256]/[25,30,256]
            point_node = self._modules['distribution2point_generation_{}'.format(l)](vi_distribution_edge, point_node)

            point_similarities.append(point_edge * self.loss_indicator[0])

            node_similarities_l2.append(node_similarity_l2 * self.loss_indicator[1])

            # distribution_similarities.append(vi_distribution_edge * self.loss_indicator[2])
            distribution_similarities.append(enhanced_edge * self.loss_indicator[2])

            sem_edge_similarities.append(sem_point_edge * self.loss_indicator[3])


        return point_similarities, node_similarities_l2, distribution_similarities, sem_edge_similarities

