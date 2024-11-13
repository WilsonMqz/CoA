import os
import json
import numpy as np
import argparse

pp = argparse.ArgumentParser(description='Format VOC 2012 metadata.')
pp.add_argument('--load-path', type=str, default='load path',
                help='Path to a directory containing a copy of the VOC dataset.')
pp.add_argument('--save-path', type=str, default='save path', help='Path to output directory.')
args = pp.parse_args()

catName_to_catID = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

catID_to_catName = {catName_to_catID[k]: k for k in catName_to_catID}

ann_dict = {}
image_list = {'train': [], 'val': []}
subsets = [{0, 1, 2, 3, 4},
           {5, 6, 7, 8, 9},
           {10, 11, 12, 13, 14},
           {15, 16, 17, 18, 19}]

for phase in ['val']:
    for cat in catName_to_catID:
        with open(os.path.join(args.load_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', cat + '_' + phase + '.txt'), 'r') as f:
            for line in f:
                cur_line = line.rstrip().split(' ')
                image_id = cur_line[0]
                label = cur_line[-1]
                image_fname = image_id + '.jpg'
                if int(label) == 1:
                    if image_fname not in ann_dict:
                        ann_dict[image_fname] = []
                        image_list[phase].append(image_fname)
                    ann_dict[image_fname].append(catName_to_catID[cat])
    # create label matrix:
    image_list[phase].sort()
    num_images = len(image_list[phase])
    label_matrix = np.zeros((num_images, len(catName_to_catID)))
    for i in range(num_images):
        cur_image = image_list[phase][i]
        label_indices = np.array(ann_dict[cur_image])
        label_matrix[i, label_indices] = 1.0
    #
    # np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels.npy'), label_matrix)
    # np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images.npy'), np.array(image_list[phase]))

    # split
    subsets0_image_list = []
    subsets0_ann_dict = {}
    subsets1_image_list = []
    subsets1_ann_dict = {}
    subsets2_image_list = []
    subsets2_ann_dict = {}
    subsets3_image_list = []
    subsets3_ann_dict = {}
    for i in range(num_images):
        cur_image = image_list[phase][i]
        label_indices = np.array(ann_dict[cur_image])
        for j in label_indices:
            if j in {0, 1, 2, 3, 4}:
                subsets0_image_list.append(cur_image)
                subsets0_ann_dict[cur_image] = ann_dict[cur_image]
                break
            if j in {5, 6, 7, 8, 9}:
                subsets1_image_list.append(cur_image)
                subsets1_ann_dict[cur_image] = ann_dict[cur_image]
                break
            if j in {10, 11, 12, 13, 14}:
                subsets2_image_list.append(cur_image)
                subsets2_ann_dict[cur_image] = ann_dict[cur_image]
                break
            if j in {15, 16, 17, 18, 19}:
                subsets3_image_list.append(cur_image)
                subsets3_ann_dict[cur_image] = ann_dict[cur_image]
                break

    subsets0_image_list.sort()
    num_images0 = len(subsets0_image_list)
    label_matrix0 = np.zeros((num_images0, len(catName_to_catID)))
    for i in range(num_images0):
        cur_image = subsets0_image_list[i]
        label_indices = np.array(subsets0_ann_dict[cur_image])
        label_matrix0[i, label_indices] = 1.0

    subsets1_image_list.sort()
    num_images1 = len(subsets1_image_list)
    label_matrix1 = np.zeros((num_images1, len(catName_to_catID)))
    for i in range(num_images1):
        cur_image = subsets1_image_list[i]
        label_indices = np.array(subsets1_ann_dict[cur_image])
        label_matrix1[i, label_indices] = 1.0

    subsets2_image_list.sort()
    num_images2 = len(subsets2_image_list)
    label_matrix2 = np.zeros((num_images2, len(catName_to_catID)))
    for i in range(num_images2):
        cur_image = subsets2_image_list[i]
        label_indices = np.array(subsets2_ann_dict[cur_image])
        label_matrix2[i, label_indices] = 1.0

    subsets3_image_list.sort()
    num_images3 = len(subsets3_image_list)
    label_matrix3 = np.zeros((num_images3, len(catName_to_catID)))
    for i in range(num_images3):
        cur_image = subsets3_image_list[i]
        label_indices = np.array(subsets3_ann_dict[cur_image])
        label_matrix3[i, label_indices] = 1.0

    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels_split0.npy'), label_matrix0)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images_split0.npy'), np.array(subsets0_image_list))

    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels_split1.npy'), label_matrix1)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images_split1.npy'), np.array(subsets1_image_list))

    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels_split2.npy'), label_matrix2)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images_split2.npy'), np.array(subsets2_image_list))

    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_labels_split3.npy'), label_matrix3)
    np.save(os.path.join(args.save_path, 'formatted_' + phase + '_images_split3.npy'), np.array(subsets3_image_list))
