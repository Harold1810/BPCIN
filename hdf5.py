import re
import torch
import csv
import h5py
import numpy as np
import json

path = '/home/public/caption/caption/coco2014/feats/COCO2014_VinVL.hdf5'

with h5py.File(path, 'r') as f:
    # print("顶级组（group）列表：")
    # print(list(f.keys()))
    print()
    vinvl_feat = f['99755_features']
    print(vinvl_feat[()])

# with open('/home/public/vqa/predictions.tsv', 'r', newline='') as file:
#     # 创建一个CSV读取器
#     reader = csv.reader(file, delimiter='\t')
#
#     # 读取并打印文件内容
#     for row in reader:
#         print(row)

# dt = np.dtype([('Tensor name', str), ('Tensor data', list)])

# 打开.tsv文件
# with open('/home/public/vqa/predictions.tsv', 'r', newline='') as file:
#     # 创建一个CSV读取器
#     reader = csv.reader(file, delimiter='\t')
#     cls = {}
#
#     # 读取并处理每行内容
#     for row in reader:
#
#         name = []
#
#         obj = row[1]
#
#         matches = re.findall(r'"class"\s*:\s*"([^"]+)"', obj, flags=re.IGNORECASE)
#         for match in matches:
#             name.append(match)
#         cls['%s' % row[0]] = name
#
#         # print(cls)
#     np.save('/home/public/caption/caption/coco2014/feats/predict', cls)

# model_state_dict = torch.load('/home/xsq/data/fsl_vqa/COCO_QA/object/train.pth')
#
# # 将模型权重参数保存为json文件
# with open('train.json', 'w') as json_file:
#     json.dump(model_state_dict, json_file)

