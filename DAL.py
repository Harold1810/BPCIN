import random
from transformers import BertTokenizer, BertModel
import os
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import BertModel, AutoModel
from torchvision import models, transforms
from torchvision.io import read_image, image
import timm
import torch.nn.functional as F
import warnings
import torch, math
import torch.nn as nn
import numpy as np
import json

class DAL_L(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dropx1 = nn.Dropout(0.1)
        # self.dropy1 = nn.Dropout(0.1)
        # self.dropx2 = nn.Dropout(0.1)
        # self.dropy2 = nn.Dropout(0.1)
        self.dropx3 = nn.Dropout(0.1)
        # self.dropy3 = nn.Dropout(0.1)

        self.attention_x = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(2)])
        # self.normx1 = nn.LayerNorm(1024, eps=1e-12)

        # self.co_atten_x = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(2)])
        # self.normx2 = nn.LayerNorm(1024, eps=1e-12)

        self.FFN_cox = nn.ModuleList([PositionWiseFeedForward() for _ in range(2)])
        # self.normx3 = nn.LayerNorm(1024, eps=1e-12)

        # self.attention_y = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(2)])
        # self.normy1 = nn.LayerNorm(1024, eps=1e-12)

        # self.co_atten_y = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(2)])
        # self.normy2 = nn.LayerNorm(1024, eps=1e-12)

        # self.FFN_coy = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(2)])
        # self.normy3 = nn.LayerNorm(1024, eps=1e-12)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, x_mask, layer_num):

        # x = self.normx1(x + self.dropx1(self.attention_x[layer_num](x, x, x, x_mask)))
        x = x + self.dropx1(self.attention_x[layer_num](x, x, x, x_mask))
        # y = self.normy1(y + self.dropy1(self.attention_y[layer_num](y, y, y, y_mask)))

        # x = self.normx2(x + self.dropx2(self.co_atten_x[layer_num](x, y, y, y_mask)))
        # y = self.normy2(y + self.dropy2(self.co_atten_y[layer_num](y, x, x, x_mask)))

        x = x + self.dropx3(self.FFN_cox[layer_num](x))
        # y = self.normy3(y + self.dropy3(self.FFN_coy[layer_num](y)))
        return x
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(1024, 1024)
        self.proj_k = nn.Linear(1024, 1024)
        self.proj_v = nn.Linear(1024, 1024)
        self.drop = nn.Dropout(0.1)
        self.scores = None
        self.n_heads = 8
    def forward(self, q, k, v, mask):
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask.float()
            scores -= 10000.0 * (1.0 - mask)
        # if scores.size(3)==20:
        #     vis_scores=scores.sum(1)[:,:,:11].sum(2)
        #     vis_scores=F.softmax(vis_scores,dim=1)
        #     print(vis_scores[0].view(7,-1))
        scores = self.drop(F.softmax(scores, dim=-1))#8,8,49,20
        
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(1024, 1024 * 4)
        self.fc2 = nn.Linear(1024 * 4, 1024)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class DAL(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dropx1 = nn.Dropout(0.1)
        self.dropy1 = nn.Dropout(0.1)
        self.dropx2 = nn.Dropout(0.1)
        self.dropy2 = nn.Dropout(0.1)
        self.dropx3 = nn.Dropout(0.1)
        self.dropy3 = nn.Dropout(0.1)

        self.attention_x = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(2)])
        self.normx1 = nn.LayerNorm(1024, eps=1e-12)

        self.co_atten_x = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(2)])
        self.normx2 = nn.LayerNorm(1024, eps=1e-12)

        self.FFN_cox = nn.ModuleList([PositionWiseFeedForward() for _ in range(2)])
        self.normx3 = nn.LayerNorm(1024, eps=1e-12)

        self.attention_y = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(2)])
        self.normy1 = nn.LayerNorm(1024, eps=1e-12)

        self.co_atten_y = nn.ModuleList([MultiHeadedSelfAttention() for _ in range(2)])
        self.normy2 = nn.LayerNorm(1024, eps=1e-12)

        self.FFN_coy = nn.ModuleList([PositionWiseFeedForward() for _ in range(2)])
        self.normy3 = nn.LayerNorm(1024, eps=1e-12)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, x_mask, y_mask, layer_num):

        # x = self.normx1(x + self.dropx1(self.attention_x[layer_num](x, x, x, x_mask)))
        # if layer_num==1:
        #     target,scores=self.attention_y[layer_num](y, y, y, y_mask)
        #     print(scores)
        #     y = self.normy1(y + self.dropy1(target))
        # y = self.normy1(y + self.dropy1(self.attention_y[layer_num](y, y, y, y_mask)))

        # x = self.normx2(x + self.dropx2(self.co_atten_x[layer_num](x, y, y, y_mask)))
        y = self.normy2(y + self.dropy2(self.co_atten_y[layer_num](y, x, x, x_mask)))

        # x = self.normx3(x + self.dropx3(self.FFN_cox[layer_num](x)))
        y = self.normy3(y + self.dropy3(self.FFN_coy[layer_num](y)))
        return x, y

class WSDAN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.blocks = DAL()
        self.n_layers = 2
        self.img_pro = nn.Linear(512,1024)
        self.q_pro = nn.Linear(512,1024)

    def forward(self, hx, hy, question_mask, img_mask=None):
        hx = self.q_pro(hx)
        hy = self.img_pro(hy)
        for i in range(self.n_layers):
            hx, hy = self.blocks(hx, hy, question_mask, img_mask, i)

        return hx,hy


