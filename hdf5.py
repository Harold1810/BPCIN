import re
import torch
import csv
import h5py
import numpy as np
import json

path = '/home/public/caption/caption/coco2014/feats/COCO2014_VinVL.hdf5'

with h5py.File(path, 'r') as f:
    # print("Top-level groups:")
    # print(list(f.keys()))
    print()
    vinvl_feat = f['99755_features']
    print(vinvl_feat[()])

# with open('/home/public/vqa/predictions.tsv', 'r', newline='') as file:
#     # Create a CSV reader.
#     reader = csv.reader(file, delimiter='\t')
#
#     # Read and print the file contents.
#     for row in reader:
#         print(row)

# dt = np.dtype([('Tensor name', str), ('Tensor data', list)])

# Open the TSV file.
# with open('/home/public/vqa/predictions.tsv', 'r', newline='') as file:
#     # Create a CSV reader.
#     reader = csv.reader(file, delimiter='\t')
#     cls = {}
#
#     # Read and process each row.
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
# # Save the model weights as a JSON file.
# with open('train.json', 'w') as json_file:
#     json.dump(model_state_dict, json_file)

