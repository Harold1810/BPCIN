import os
import logging
import torch
import shutil

# 初始化 训练参数
def allocate_tensors():
    """
    init data tensors
    :return: data tensors
    """
    tensors = dict()
    tensors['support_data'] = torch.FloatTensor()
    tensors['support_que'] = torch.IntTensor()
    tensors['support_label'] = torch.LongTensor()
    tensors['query_data'] = torch.FloatTensor()
    tensors['query_que'] = torch.IntTensor()
    tensors['query_label'] = torch.LongTensor()
    tensors['support_vinvl'] = torch.FloatTensor()
    tensors['query_vinvl'] = torch.FloatTensor()
    tensors['support_cls'] = torch.IntTensor()
    tensors['query_cls'] = torch.IntTensor()
    # tensors['task_vectors'] = torch.FloatTensor()

    return tensors

# 将 初始化参数 赋值给 需要训练的参数
def set_tensors(tensors, batch):
    """
    set data to initialized tensors
    :param tensors: initialized data tensors
    :param batch: current batch of data
    :return: None
    """
    support_data, support_que, support_label, query_data, query_que, query_label, support_vinvl, query_vinvl,\
    support_cls, query_cls = batch
    tensors['support_data'].resize_(support_data.size()).copy_(support_data)
    tensors['support_que'].resize_(support_que.size()).copy_(support_que)
    tensors['support_label'].resize_(support_label.size()).copy_(support_label)
    tensors['query_data'].resize_(query_data.size()).copy_(query_data)
    tensors['query_que'].resize_(query_que.size()).copy_(query_que)
    tensors['query_label'].resize_(query_label.size()).copy_(query_label)
    tensors['support_vinvl'].resize_(support_vinvl.size()).copy_(support_vinvl)
    tensors['query_vinvl'].resize_(query_vinvl.size()).copy_(query_vinvl)
    tensors['support_cls'].resize_(support_cls.size()).copy_(support_cls)
    tensors['query_cls'].resize_(query_cls.size()).copy_(query_cls)
    # tensors['task_vectors'].resize_(task_vectors.size()).copy_(task_vectors)


def set_logging_config(logdir):
    """
    set logging configuration
    :param logdir: directory put logs
    :return: None
    """
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if DPGN model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'),
                        os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))


def adjust_learning_rate(optimizers, lr, iteration, dec_lr_step, lr_adj_base):
    """
    adjust learning rate after some iterations
    :param optimizers: the optimizers
    :param lr: learning rate
    :param iteration: current iteration
    :param dec_lr_step: decrease learning rate in how many step
    :return: None
    """
    new_lr = lr * (lr_adj_base ** (int(iteration / dec_lr_step)))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

# 将标签转换为边信息
def label2edge(label, device):
    """
    convert ground truth labels into ground truth edges
    :param label: ground truth labels
    :param device: the gpu device that holds the ground truth edges
    :return: ground truth edges
    """
    # get size
    num_samples = label.size(1)
    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)
    # compute edge
    edge = torch.eq(label_i, label_j).float().to(device)
    return edge


def one_hot_encode(num_classes, class_idx, device):
    """
    one-hot encode the ground truth
    :param num_classes: number of total class
    :param class_idx: belonging class's index
    :param device: the gpu device that holds the one-hot encoded ground truth label
    :return: one-hot encoded ground truth label
    """
    return torch.eye(num_classes)[class_idx.cpu()].to(device)

# 生成 support 与 query 的 mask
def preprocessing(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots  # 5*1
    num_samples = num_supports + num_queries * num_ways # 5 + 1*5

    # set edge mask (to distinguish support and query edges)
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)  # (25,10,10)
    support_edge_mask[:, :num_supports, :num_supports] = 1  # 就是四等分 10*10的矩阵 左上角全部都是1
    # print(support_edge_mask)
    query_edge_mask = 1 - support_edge_mask  # 与 support_edge_mask 正好相反
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)  # eval 全1

    return num_supports, num_samples, query_edge_mask, evaluation_mask

# 初始化 边的权重
def initialize_nodes_edges(batch, num_supports, tensors, batch_size, num_queries, num_ways, device):
    # allocate data in this batch to specific variables
    set_tensors(tensors, batch)
    support_data = tensors['support_data'].squeeze(0)    # [25,5,3,84,84]
    support_que = tensors['support_que'].squeeze(0)
    support_label = tensors['support_label'].squeeze(0)  # [25,5]
    query_data = tensors['query_data'].squeeze(0)        # [25,5,3,84,84]
    query_que = tensors['query_que'].squeeze(0)
    query_label = tensors['query_label'].squeeze(0)      # [25,5]
    support_vinvl = tensors['support_vinvl'].squeeze(0)
    query_vinvl = tensors['query_vinvl'].squeeze(0)
    support_cls = tensors['support_cls'].squeeze(0)
    query_cls = tensors['query_cls'].squeeze(0)
    # task_vectors = tensors['task_vectors'].squeeze(0)

    # initialize nodes of distribution graph
    node_gd_init_support = label2edge(support_label, device)  # [25,5,5] 这里其实就是25个 5*5 的单位矩阵  初始化 support 对应的distribution点
    node_gd_init_query = (torch.ones([batch_size, num_queries * num_ways, num_supports])
                          * torch.tensor(1. / num_supports)).to(device)  # [25,5,5] 初始化 query 对应 的distribution 点
    node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)  # [25,10,5]  可能是因为只需要判断每个顶点与 support的关系
                                                                                    # 所以这里只是10*5 而不是 10*10
    # initialize edges of point graph
    all_data = torch.cat([support_data, query_data], 1)  # [25,10,3,84,84]
    all_que = torch.cat([support_que, query_que], 1)  # [25,10,15]
    all_label = torch.cat([support_label, query_label], 1)  # [25,10]
    all_label_in_edge = label2edge(all_label, device)  # [25,10,10]  单位阵

    all_vinvl = torch.cat([support_vinvl, query_vinvl], 1)
    all_cls = torch.cat([support_cls, query_cls], 1)

    return support_data, support_que, support_label, query_data, query_que, query_label, all_data, all_que, all_label_in_edge, \
           node_feature_gd, all_vinvl, all_cls

# 提取特征
def backbone_two_stage_initialization(full_data, encoder):
    """
    encode raw data by backbone network
    :param full_data: raw data
    :param encoder: backbone network
    :return: last layer logits from backbone network
             second last layer logits from backbone network
    """
    # encode data
    last_layer_data_temp = []
    second_last_layer_data_temp = []
    for data in full_data.chunk(full_data.size(1), dim=1):
        # the encode step  data:[25,1,3,84,84]
        encoded_result = encoder(data.squeeze(1))  # 两个 [25,128]  coco:[10,128] 最后一层和倒数第二层特征
        # prepare for two stage initialization of DPGN
        last_layer_data_temp.append(encoded_result[0])
        second_last_layer_data_temp.append(encoded_result[1])
    # last_layer_data: (batch_size, num_samples, embedding dimension)
    last_layer_data = torch.stack(last_layer_data_temp, dim=1)  # [25,10,128] 将temp里的特征在新的维度拼接在一起
    # second_last_layer_data: (batch_size, num_samples, embedding dimension)
    second_last_layer_data = torch.stack(second_last_layer_data_temp, dim=1)  # [25,10,128]
    return last_layer_data, second_last_layer_data


# 提取特征
def ques_initialization(full_data, cls_data, encoder1, encoder2):
    """
    encode raw data by backbone network
    :param full_data: raw data
    :param encoder: backbone network
    :return: last layer logits from backbone network
             second last layer logits from backbone network
    """
    # encode data
    # full_data:[10,10,15] cls_data:[10,10,50,300]
    last_que_temp = []
    que_embedding_temp = []
    cls_temp = []
    cls_embedding_temp = []
    for data in full_data.chunk(full_data.size(1), dim=1):
        # the encode step  data:[25,1,15]
        que_feat, que_embedding = encoder1(data.squeeze(1))
        last_que_temp.append(que_feat)
        que_embedding_temp.append(que_embedding)

    for cls in cls_data.chunk(cls_data.size(1), dim=1):
        encoded_result, cls_embedding = encoder2(cls.squeeze(1))
        cls_temp.append(encoded_result)
        cls_embedding_temp.append(cls_embedding)

    # last_layer_data: (batch_size, num_samples, embedding dimension)
    last_data = torch.stack(last_que_temp, dim=1)
    que_embedding_last = torch.stack(que_embedding_temp, dim=1)
    cls = torch.stack(cls_temp, dim=1)
    cls_embedding_last = torch.stack(cls_embedding_temp, dim=1)
    # second_last_layer_data: (batch_size, num_samples, embedding dimension)
    return last_data, cls, que_embedding_last, cls_embedding_last

