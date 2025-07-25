import clip

from backbone import ResNet12, ConvNet, Clip
from question_encoder import Question_Encoder, Cls_Encoder
from dpgn import DPGN
from utils import set_logging_config, adjust_learning_rate, save_checkpoint, allocate_tensors, preprocessing, \
    initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode, ques_initialization
from dataloader import FSL_VQA, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import logging
import argparse
import imp

torch.autograd.set_detect_anomaly(True)

class DPGNTrainer(object):
    def __init__(self, enc_module, gnn_module, que_enc, data_loader, log, arg, config, best_step, cls_enc):
        """
        The Trainer of DPGN model
        :param enc_module: backbone network (Conv4, ResNet12, ResNet18, WRN)
        :param gnn_module: DPGN model
        :param data_loader: data loader
        :param log: logger
        :param arg: command line arguments
        :param config: model configurations
        :param best_step: starting step (step at best eval acc or 0 if starts from scratch)
        """

        self.arg = arg  # 当前轮次相关参数
        self.config = config  # 训练相关参数
        self.train_opt = config['train_config']  # 训练参数
        self.eval_opt = config['eval_config']  # 测试验证参数

        # initialize variables
        self.tensors = allocate_tensors()  # 初始化训练参数
        for key, tensor in self.tensors.items():
            self.tensors[key] = tensor.to(self.arg.device)

        # set backbone and DPGN
        self.enc_module = enc_module.to(arg.device)
        self.enc_que_module = que_enc.to(arg.device)
        self.gnn_module = gnn_module.to(arg.device)
        self.cls_enc = cls_enc.to(arg.device)

        # set logger
        self.log = log

        # get data loader
        self.data_loader = data_loader

        # set parameters
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters()) + list(self.enc_que_module.parameters())

        # set optimizer
        self.optimizer = optim.Adam(
            params=self.module_params,
            lr=self.train_opt['lr'],  # 0.001
            weight_decay=self.train_opt['weight_decay'])  # 1e.-5   

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.pred_loss = nn.CrossEntropyLoss(reduction='none')

        # initialize other global variables
        self.global_step = best_step
        self.best_step = best_step
        self.val_acc = 0
        self.test_acc = 0

    def train(self):
        """
        train function
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = \
            preprocessing(self.train_opt['num_ways'],  # 5
                          self.train_opt['num_shots'],  # 1
                          self.train_opt['num_queries'],  # 1
                          self.train_opt['batch_size'],  # 25
                          self.arg.device)

        # main training loop, batch size is the number of tasks
        for iteration, batch in enumerate(self.data_loader['train']()):
            # init grad
            self.optimizer.zero_grad()  # 梯度清零

            # set current step
            self.global_step += 1

            # initialize nodes and edges for dual graph model
            support_data, support_que, support_label, query_data, query_que, query_label, all_data, all_que, \
            all_label_in_edge, node_feature_gd, all_vinvl, all_cls = initialize_nodes_edges(batch,
                                                                      num_supports,  # 5
                                                                      self.tensors,  # 初始化的 训练参数
                                                                      self.train_opt['batch_size'],   # 25
                                                                      self.train_opt['num_queries'],  # 1
                                                                      self.train_opt['num_ways'],     # 5
                                                                      self.arg.device)

            # set as train mode
            self.enc_module.train()
            self.enc_que_module.train()
            self.gnn_module.train()
            self.cls_enc.train()

            # use backbone encode image
            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            que_data, cls_data, que_embedding, cls_embedding = ques_initialization(all_que, all_cls, self.enc_que_module, self.cls_enc)

            # run the DPGN model
            point_similarity, node_similarity_l2, distribution_similarities, yuyi_edge_similarities = self.gnn_module(second_last_layer_data,
                                                                                                 last_layer_data,
                                                                                                 que_data,
                                                                                                 node_feature_gd,
                                                                                                 all_vinvl,
                                                                                                 cls_data,
                                                                                                 que_embedding,
                                                                                                 cls_embedding)

            # compute loss
            total_loss, query_node_cls_acc_generations, query_edge_loss_generations = \
                self.compute_train_loss_pred(all_label_in_edge,   # [25,10,10]
                                             point_similarity,    # 6个 [25,10,10]
                                             node_similarity_l2,  # 6个 [25,10,10]
                                             query_edge_mask,     # [25,10,10]
                                             evaluation_mask,     # [25,10,10]
                                             num_supports,        # 5
                                             support_label,       # [25,5]
                                             query_label,         # [25,5]
                                             distribution_similarities, # 6个 [25,10,10]
                                             yuyi_edge_similarities)


            # back propagation & update
            total_loss.backward()
            self.optimizer.step()

            # adjust learning rate
            adjust_learning_rate(optimizers=[self.optimizer],
                                 lr=self.train_opt['lr'],
                                 iteration=self.global_step,
                                 dec_lr_step=self.train_opt['dec_lr'],
                                 lr_adj_base =self.train_opt['lr_adj_base'])

            # log training info
            if self.global_step % self.arg.log_step == 0:
                self.log.info('step : {}  train_edge_loss : {}  node_acc : {}'.format(
                    self.global_step,
                    query_edge_loss_generations[-1],
                    query_node_cls_acc_generations[-1]))

            # evaluation
            if self.global_step % self.eval_opt['interval'] == 0:
                is_best = 0
                test_acc = self.eval(partition='test')
                if test_acc > self.test_acc:
                    is_best = 1
                    self.test_acc = test_acc
                    self.best_step = self.global_step

                # log evaluation info
                self.log.info('test_acc : {}         step : {} '.format(test_acc, self.global_step))
                self.log.info('test_best_acc : {}    step : {}'.format( self.test_acc, self.best_step))

                # save checkpoints (best and newest)
                save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'que_enc_module_state_dict': self.enc_que_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'test_acc': self.test_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'enc_cls':self.cls_enc.state_dict(),
                    'cls_enc_module_state_dict':self.cls_enc.state_dict()
                }, is_best, os.path.join(self.arg.checkpoint_dir, self.arg.exp_name))

    def eval(self, partition='test', log_flag=True):
        """
        evaluation function
        :param partition: which part of data is used
        :param log_flag: if log the evaluation info
        :return: None
        """

        num_supports, num_samples, query_edge_mask, evaluation_mask = preprocessing(
            self.eval_opt['num_ways'],
            self.eval_opt['num_shots'],
            self.eval_opt['num_queries'],
            self.eval_opt['batch_size'],
            self.arg.device)

        query_edge_loss_generations = []
        query_node_cls_acc_generations = []
        # main training loop, batch size is the number of tasks
        for current_iteration, batch in enumerate(self.data_loader[partition]()):
            # initialize nodes and edges for dual graph model
            support_data, support_que, support_label, query_data, query_que, query_label, all_data, all_que, \
            all_label_in_edge, node_feature_gd, all_vivnl, all_cls = initialize_nodes_edges(batch,
                                                                      num_supports,
                                                                      self.tensors,
                                                                      self.eval_opt['batch_size'],
                                                                      self.eval_opt['num_queries'],
                                                                      self.eval_opt['num_ways'],
                                                                      self.arg.device)

            # set as eval mode
            self.enc_module.eval()
            self.enc_que_module.eval()
            self.gnn_module.eval()
            self.cls_enc.eval()

            last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, self.enc_module)

            que_data, cls_data,que_embedding, cls_embedding = ques_initialization(all_que, all_cls, self.enc_que_module, self.cls_enc)

            # run the DPGN model
            point_similarity, _, _, _ = self.gnn_module(second_last_layer_data,
                                                     last_layer_data,
                                                     que_data,
                                                     node_feature_gd,
                                                     all_vivnl,
                                                     cls_data,
                                                     que_embedding,
                                                     cls_embedding)

            query_node_cls_acc_generations, query_edge_loss_generations = \
                self.compute_eval_loss_pred(query_edge_loss_generations,
                                            query_node_cls_acc_generations,
                                            all_label_in_edge,
                                            point_similarity,
                                            query_edge_mask,
                                            evaluation_mask,
                                            num_supports,
                                            support_label,
                                            query_label)

        # logging
        if log_flag:
            self.log.info('------------------------------------')
            self.log.info('step : {}  {}_edge_loss : {}  {}_node_acc : {}'.format(
                self.global_step, partition,
                np.array(query_edge_loss_generations).mean(),
                partition,
                np.array(query_node_cls_acc_generations).mean()))

            self.log.info('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                          (current_iteration,
                           np.array(query_node_cls_acc_generations).mean() * 100,
                           np.array(query_node_cls_acc_generations).std() * 100,
                           1.96 * np.array(query_node_cls_acc_generations).std()
                           / np.sqrt(float(len(np.array(query_node_cls_acc_generations)))) * 100))
            self.log.info('------------------------------------')

        return np.array(query_node_cls_acc_generations).mean()

    def compute_train_loss_pred(self,
                                all_label_in_edge,
                                point_similarities,
                                node_similarities_l2,
                                query_edge_mask,
                                evaluation_mask,
                                num_supports,
                                support_label,
                                query_label,
                                distribution_similarities,
                                yuyi_edge_similarities
                                ):
        """
        compute the total loss, query classification loss and query classification accuracy
        :param all_label_in_edge: ground truth label in edge form of point graph
        :param point_similarities: prediction edges of point graph
        :param node_similarities_l2: l2 norm of node similarities
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :param distribution_similarities: distribution-level similarities
        :return: total loss
                 query classification accuracy
                 query classification loss
        """

        # Point Loss
        total_edge_loss_generations_instance = [
            self.edge_loss((1 - point_similarity), (1 - all_label_in_edge))
            for point_similarity
            in point_similarities]

        total_yuyi_loss_generations_instance = [
            self.edge_loss((1 - yuyi_edge_similarity), (1 - all_label_in_edge))
            for yuyi_edge_similarity
            in yuyi_edge_similarities]

        # Distribution Loss
        total_edge_loss_generations_distribution = [
            self.edge_loss((1 - distribution_similarity), (1 - all_label_in_edge))
            for distribution_similarity
            in distribution_similarities]

        # combine Point Loss and Distribution Loss
        distribution_loss_coeff = 0.1
        yuyi_rate = 0.1
        total_edge_loss_generations = [
            total_edge_loss_instance + distribution_loss_coeff * total_edge_loss_distribution + yuyi_rate * total_yuyi_edge_loss
            for (total_edge_loss_instance, total_edge_loss_distribution, total_yuyi_edge_loss)
            in zip(total_edge_loss_generations_instance, total_edge_loss_generations_distribution, total_yuyi_loss_generations_instance)]

        pos_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * all_label_in_edge * evaluation_mask)
            / torch.sum(query_edge_mask * all_label_in_edge * evaluation_mask)
            for total_edge_loss_generation
            in total_edge_loss_generations]

        neg_query_edge_loss_generations = [
            torch.sum(total_edge_loss_generation * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)
            / torch.sum(query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)
            for total_edge_loss_generation
            in total_edge_loss_generations]

        # weighted edge loss for balancing pos/neg
        query_edge_loss_generations = [
            pos_query_edge_loss_generation + neg_query_edge_loss_generation
            for (pos_query_edge_loss_generation, neg_query_edge_loss_generation)
            in zip(pos_query_edge_loss_generations, neg_query_edge_loss_generations)]

        # (normalized) l2 loss
        query_node_pred_generations_ = [
            torch.bmm(node_similarity_l2[:, num_supports:, :num_supports],
                      one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for node_similarity_l2
            in node_similarities_l2]

        # prediction
        query_node_pred_generations = [
            torch.bmm(point_similarity[:, num_supports:, :num_supports],
                      one_hot_encode(self.train_opt['num_ways'], support_label.long(), self.arg.device))
            for point_similarity
            in point_similarities]

        query_node_pred_loss = [
            self.pred_loss(query_node_pred_generation, query_label.long()).mean()
            for query_node_pred_generation
            in query_node_pred_generations_]

        # train accuracy
        query_node_acc_generations = [
            torch.eq(torch.max(query_node_pred_generation, -1)[1], query_label.long()).float().mean()
            for query_node_pred_generation
            in query_node_pred_generations]

        # total loss
        total_loss_generations = [
            query_edge_loss_generation + 0.1 * query_node_pred_loss_
            for (query_edge_loss_generation, query_node_pred_loss_)
            in zip(query_edge_loss_generations, query_node_pred_loss)]

        # compute total loss
        total_loss = []
        num_loss = self.config['num_loss_generation']
        for l in range(num_loss - 1):
            total_loss += [total_loss_generations[l].view(-1) * self.config['generation_weight']]
        total_loss += [total_loss_generations[-1].view(-1) * 1.0]
        total_loss = torch.mean(torch.cat(total_loss, 0))

        # with open('log.txt', 'a') as f:
        #     for i in range(len(point_similarities)):
        #         f.write(f"predicted label: {point_similarities[i]}, true label: {all_label_in_edge[i]}\n")

        return total_loss, query_node_acc_generations, query_edge_loss_generations

    def compute_eval_loss_pred(self,
                               query_edge_losses,
                               query_node_accs,
                               all_label_in_edge,
                               point_similarities,
                               query_edge_mask,
                               evaluation_mask,
                               num_supports,
                               support_label,
                               query_label):
        """
        compute the query classification loss and query classification accuracy
        :param query_edge_losses: container for losses of queries' edges
        :param query_node_accs: container for classification accuracy of queries
        :param all_label_in_edge: ground truth label in edge form of point graph
        :param point_similarities: prediction edges of point graph
        :param query_edge_mask: mask for queries
        :param evaluation_mask: mask for evaluation (for unsupervised setting)
        :param num_supports: number of samples in support set
        :param support_label: label of support set
        :param query_label: label of query set
        :return: query classification loss
                 query classification accuracy
        """

        point_similarity = point_similarities[-1]
        full_edge_loss = self.edge_loss(1 - point_similarity, 1 - all_label_in_edge)

        pos_query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * all_label_in_edge * evaluation_mask) / torch.sum(
            query_edge_mask * all_label_in_edge * evaluation_mask)
        neg_query_edge_loss = torch.sum(
            full_edge_loss * query_edge_mask * (1 - all_label_in_edge) * evaluation_mask) / torch.sum(
            query_edge_mask * (1 - all_label_in_edge) * evaluation_mask)

        # weighted loss for balancing pos/neg
        query_edge_loss = pos_query_edge_loss + neg_query_edge_loss

        # prediction
        query_node_pred = torch.bmm(
            point_similarity[:, num_supports:, :num_supports],
            one_hot_encode(self.eval_opt['num_ways'], support_label.long(), self.arg.device))

        # test accuracy
        query_node_acc = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

        query_edge_losses += [query_edge_loss.item()]
        query_node_accs += [query_node_acc.item()]

        return query_node_accs, query_edge_losses


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:2',
                        help='gpu device number of using')

    parser.add_argument('--config', type=str, default=os.path.join('.', 'config', '5way_5shot_clip_coco_qa.py'),
                        help='config file with parameters of the experiment. '
                             'It is assumed that the config file is placed under the directory ./config')

    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('.', 'fpait_checkpoints'),
                        help='path that checkpoint will be saved and loaded. '
                             'It is assumed that the checkpoint file is placed under the directory ./checkpoints')

    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu')

    parser.add_argument('--display_step', type=int, default=100,
                        help='display training information in how many step')

    parser.add_argument('--log_step', type=int, default=100,
                        help='log information in how many steps')

    parser.add_argument('--log_dir', type=str, default=os.path.join('.', 'fpait_logs'),
                        help='path that log will be saved. '
                             'It is assumed that the checkpoint file is placed under the directory ./logs')

    parser.add_argument('--dataset_root', type=str, default='/home/xsq/data/fsl_vqa/COCO_QA',
                        help='root directory of dataset')

    parser.add_argument('--seed', type=int, default=222,
                        help='random seed')

    parser.add_argument('--mode', type=str, default='train',
                        help='train or eval')

    args_opt = parser.parse_args()

    config_file = args_opt.config

    # Set train and test datasets and the corresponding data loaders
    config = imp.load_source("", config_file).config
    train_opt = config['train_config']   # 训练的设置参数
    eval_opt = config['eval_config']     # 测试的设置参数

    args_opt.exp_name = '{}way_{}shot_{}_{}'.format(train_opt['num_ways'],
                                                    train_opt['num_shots'],
                                                    config['backbone'],
                                                    config['dataset_name'])
    train_opt['num_queries'] = 1
    eval_opt['num_queries'] = 1
    set_logging_config(os.path.join(args_opt.log_dir, args_opt.exp_name))
    logger = logging.getLogger('main')

    # Load the configuration params of the experiment
    logger.info('Launching experiment from: {}'.format(config_file))
    logger.info('Generated logs will be saved to: {}'.format(args_opt.log_dir))
    logger.info('Generated checkpoints will be saved to: {}'.format(args_opt.checkpoint_dir))
    print()

    logger.info('-------------command line arguments-------------')
    logger.info(args_opt)
    print()
    logger.info('-------------configs-------------')
    logger.info(config)

    # set random seed
    np.random.seed(args_opt.seed)
    torch.manual_seed(args_opt.seed)
    torch.cuda.manual_seed_all(args_opt.seed)
    random.seed(args_opt.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = FSL_VQA
    print('Dataset: {}'.format(config['dataset_name']))

    dataset_train = dataset(root=args_opt.dataset_root, category=config['dataset_name'], partition='train')
    # dataset_train = dataset(root=args_opt.dataset_root, category=config['dataset_name'], partition='train_fpait')
    # dataset_valid = dataset(root=args_opt.dataset_root, partition='val', token_to_ix=dataset_train.token_to_ix)
    dataset_test = dataset(root=args_opt.dataset_root, category=config['dataset_name'], partition='test', token_to_ix=dataset_train.token_to_ix)
    # dataset_test = dataset(root=args_opt.dataset_root, category=config['dataset_name'], partition='test_fpait', token_to_ix=dataset_train.token_to_ix)

    train_loader = DataLoader(dataset_train,
                              num_tasks=train_opt['batch_size'],  # 25
                              num_ways=train_opt['num_ways'],     # 5
                              num_shots=train_opt['num_shots'],   # 1
                              num_queries=train_opt['num_queries'],  # 1
                              epoch_size=train_opt['iteration'])  # 100000
    # valid_loader = DataLoader(dataset_valid,
    #                           num_tasks=eval_opt['batch_size'],  # 10
    #                           num_ways=eval_opt['num_ways'],     # 5
    #                           num_shots=eval_opt['num_shots'],   # 1
    #                           num_queries=eval_opt['num_queries'],  # 1
    #                           epoch_size=eval_opt['iteration'])  # 1000
    test_loader = DataLoader(dataset_test,
                             num_tasks=eval_opt['batch_size'],   # 10
                             num_ways=eval_opt['num_ways'],      # 5
                             num_shots=eval_opt['num_shots'],    # 1
                             num_queries=eval_opt['num_queries'],  # 1
                             epoch_size=eval_opt['iteration'])   # 1000

    data_loader = {'train': train_loader,
                #    'val': valid_loader,
                   'test': test_loader}

    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'])
        print('Backbone: ResNet12')
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'])
        print('Backbone: ConvNet')
    elif config['backbone'] == 'clip':
        enc_module = Clip(emb_size=config['emb_size'], args=args_opt)
    else:
        logger.info('Invalid backbone: {}, please specify a backbone model from '
                    'convnet or resnet12.'.format(config['backbone']))
        exit()

    # 创建DPGN模块
    que_enc = Question_Encoder(dataset_train.pretrained_emb, dataset_train.token_size)
    cls_enc = Cls_Encoder(dataset_train.pretrained_emb, dataset_train.token_size)

    gnn_module = DPGN(config['num_generation'],  # 6
                      train_opt['dropout'],      # 0.1
                      train_opt['num_ways'],  # 5
                      train_opt['num_shots'],  # 1/5
                      train_opt['num_ways'] * train_opt['num_shots'],  # 5 * 1
                      train_opt['num_ways'] * train_opt['num_shots'] + train_opt['num_ways'],  # 5*1 + 5*1
                      train_opt['loss_indicator'],  # [1,1,0]
                      config['point_distance_metric'],  # 'l2'
                      config['distribution_distance_metric']   # 'l2'
                      )

    # multi-gpu configuration
    [print('GPU: {}  Spec: {}'.format(i, torch.cuda.get_device_name(i))) for i in range(args_opt.num_gpu)]

    if args_opt.num_gpu > 1:
        print('Construct multi-gpu model ...')
        enc_module = nn.DataParallel(enc_module, device_ids=range(args_opt.num_gpu), dim=0)
        gnn_module = nn.DataParallel(gnn_module, device_ids=range(args_opt.num_gpu), dim=0)
        print('done!\n')

    if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)):
        os.makedirs(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name))
        logger.info('no checkpoint for model: {}, make a new one at {}'.format(
            args_opt.exp_name,
            os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
        best_step = 0
    else:
        if not os.path.exists(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar')):
            best_step = 0
        else:
            logger.info('find a checkpoint, loading checkpoint from {}'.format(
                os.path.join(args_opt.checkpoint_dir, args_opt.exp_name)))
            best_checkpoint = torch.load(os.path.join(args_opt.checkpoint_dir, args_opt.exp_name, 'model_best.pth.tar'))

            logger.info('best model pack loaded')
            best_step = best_checkpoint['iteration']
            enc_module.load_state_dict(best_checkpoint['enc_module_state_dict'])
            que_enc.load_state_dict(best_checkpoint['que_enc_module_state_dict'])
            gnn_module.load_state_dict(best_checkpoint['gnn_module_state_dict'])
            cls_enc.load_state_dict(best_checkpoint['cls_enc_module_state_dict'])
            logger.info('current best test accuracy is: {}, at step: {}'.format(best_checkpoint['test_acc'], best_step))


    # create trainer
    trainer = DPGNTrainer(enc_module=enc_module,  # encoder
                           gnn_module=gnn_module,  # DPGN
                           que_enc=que_enc,
                           data_loader=data_loader,  # data_loader
                           log=logger,
                           arg=args_opt,
                           config=config,
                           best_step=best_step,
                          cls_enc=cls_enc)

    if args_opt.mode == 'train':
        trainer.train()
    elif args_opt.mode == 'eval':
        trainer.eval()
    else:
        print('select a mode')
        exit()


if __name__ == '__main__':
    main()
