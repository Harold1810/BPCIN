import torch

# lineidx_file = "/home/public/vqa/VinVL_output/inference/vinvl_vg_x152c4_1/predictions.lineidx"
# with open(lineidx_file, 'r') as fpidx:
#     lineidx = fpidx.read()
#     img_id_list = lineidx.split('\n')[:-1]
#     print(len(img_id_list))

predictions = torch.load("/home/public/vqa/VinVL_output/inference/vinvl_vg_x152c4_1/predictions.pth")
# print(len(predictions))
for idx, boxlist in predictions.items():

    print(idx, boxlist)  # 0 BoxList(num_boxes=65, image_width=800, image_height=600, mode=xyxy)
    # print(boxlist.extra_fields.keys())
    # dict_keys(['labels', 'scores', 'box_features', 'scores_all', 'boxes_all', 'attr_labels', 'attr_scores'])
    # print("img_id:", img_id_list[idx])
    print("feature:", boxlist.extra_fields['box_features'].shape)
    print("label:", boxlist.extra_fields['labels'].shape)
    print()
    # break


