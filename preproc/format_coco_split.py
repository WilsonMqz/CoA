import json
import os
import argparse
import numpy as np

pp = argparse.ArgumentParser(description='Format COCO metadata.')
pp.add_argument('--load-path', type=str, default='load path',
                help='Path to a directory containing a copy of the COCO dataset.')
pp.add_argument('--save-path', type=str, default='save path', help='Path to output directory.')
args = pp.parse_args()

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

# initialize metadata dictionary:
meta = {}
meta['category_id_to_index'] = {}
meta['category_list'] = []

for split in ['val']:
    
    with open(os.path.join(args.load_path, 'annotations', 'instances_' + split + '2014.json'), 'r') as f:
        D = json.load(f)
    
    if len(meta['category_list']) == 0:
        # parse the category data:
        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
    else:
        # check that category lists are consistent for train2014 and val2014:
        (category_list, id_to_index) = parse_categories(D['categories'])
        assert category_list == meta['category_list']
        assert id_to_index == meta['category_id_to_index']

    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    # sorting as strings for backwards compatibility 
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}
    
    num_categories = len(D['categories'])
    num_images = len(image_id_list)
    
    label_matrix = np.zeros((num_images,num_categories))
    image_ids = np.zeros(num_images)
    
    for i in range(len(D['annotations'])):
        
        image_id = int(D['annotations'][i]['image_id'])
        row_index = image_id_to_index[image_id]
    
        category_id = int(D['annotations'][i]['category_id'])
        category_index = int(meta['category_id_to_index'][category_id])
        
        label_matrix[row_index][category_index] = 1
        image_ids[row_index] = int(image_id)
    
    # image_ids = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids])
    
    # save labels and corresponding image ids: 
    # np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), label_matrix)
    # np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), image_ids)

# save metadata: 
# with open(os.path.join(args.save_path, 'annotations', 'formatted_metadata.json'), 'w') as f:
#     json.dump(meta, f)
    num_images0 = 0
    num_images1 = 0
    num_images2 = 0
    num_images3 = 0
    for i in range(len(image_ids)):
        cur_label_matrix = label_matrix[i]
        index_list = [i for i, value in enumerate(cur_label_matrix) if value == 1]
        for j in index_list:
            if j in range(0, 20):
                num_images0 += 1
                break
            if j in range(20, 40):
                num_images1 += 1
                break
            if j in range(40, 60):
                num_images2 += 1
                break
            if j in range(60, 80):
                num_images3 += 1
                break

    label_matrix0 = np.zeros((num_images0, num_categories))
    label_matrix1 = np.zeros((num_images1, num_categories))
    label_matrix2 = np.zeros((num_images2, num_categories))
    label_matrix3 = np.zeros((num_images3, num_categories))
    image_ids0 = np.zeros(num_images0)
    image_ids1 = np.zeros(num_images1)
    image_ids2 = np.zeros(num_images2)
    image_ids3 = np.zeros(num_images3)
    index0 = 0
    index1 = 0
    index2 = 0
    index3 = 0
    for i in range(len(image_ids)):
        cur_label_matrix = label_matrix[i]
        index_list = [i for i, value in enumerate(cur_label_matrix) if value == 1]
        for j in index_list:
            if j in range(0, 20):
                label_matrix0[index0] = cur_label_matrix
                image_ids0[index0] = image_ids[i]
                index0 = index0 + 1
                break
            if j in range(20, 40):
                label_matrix1[index1] = cur_label_matrix
                image_ids1[index1] = image_ids[i]
                index1 = index1 + 1
                break
            if j in range(40, 60):
                label_matrix2[index2] = cur_label_matrix
                image_ids2[index2] = image_ids[i]
                index2 = index2 + 1
                break
            if j in range(60, 80):
                label_matrix3[index3] = cur_label_matrix
                image_ids3[index3] = image_ids[i]
                index3 = index3 + 1
                break
    image_ids0 = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids0])
    image_ids1 = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids1])
    image_ids2 = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids2])
    image_ids3 = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids3])

    # save labels and corresponding image ids:
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels_split0.npy'), label_matrix0)
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images_split0.npy'), image_ids0)

    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels_split1.npy'), label_matrix1)
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images_split1.npy'), image_ids1)

    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels_split2.npy'), label_matrix2)
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images_split2.npy'), image_ids2)

    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels_split3.npy'), label_matrix3)
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images_split3.npy'), image_ids3)