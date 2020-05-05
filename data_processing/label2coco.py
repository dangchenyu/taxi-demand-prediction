import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def init_coco_dict():
    return {
        'info': {
            'description': 'Kyoto pedestrian dataset',
            'url': 'https://www.vision.rwth-aachen.de/page/mots',
            'version': '1.0',
            'year': 2020,
            'contributor': 'chenyu',
            'date_created': '2020/5/2'
        },
        'licenses': [
            {
                'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
                'id': 1,
                'name': 'Attribution-NonCommercial-ShareAlike License'
            }
        ],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'supercategory': 'person',
                'id': 1,
                'name': 'person'
            }
        ]
    }


def to_coco(root_path):
    train_dict = init_coco_dict()
    val_dict = init_coco_dict()

    image_count = 0
    instance_count = 0
    cut_y=80
    images_path = os.path.join(root_path, 'raw_frames')
    label_path = os.path.join(root_path, 'annotations', 'raw_labels')
    labels = os.listdir(label_path)
    labels.sort()

    for label in labels:
        id_pool = []
        print('Processing label {}'.format(label))
        o = open(os.path.join(label_path, label), 'r')
        label_base_name = os.path.splitext(label)[0]
        lines = o.readlines()
        imgs_num = len(lines)

        for line in lines:
            line_list = line.split(',')
            frame_num = line_list[0]
            image_count += 1
            frame_img = label_base_name + '_{:06d}.jpg'.format(int(frame_num))

            image_dict = {
                'license': 1,
                'file_name': frame_img,
                'coco_url': '',
                'height': 640,
                'width': 640,
                'date_captured': '',
                'flickr_url': '',
                'id': image_count
            }
            # train_dict['images'].append(image_dict)
            if image_count >= 0.8 * imgs_num:
                val_dict['images'].append(image_dict)
            else:
                train_dict['images'].append(image_dict)
            obj_num = line_list[1]
            for obj in range(int(obj_num)):
                print(obj)
                instance_count += 1
                bbox = line_list[3 + 6 * obj:7 + 6 * obj]
                instance_dict = {
                    'iscrowd': 0,
                    'image_id': image_count,
                    'category_id': 1,
                    'id': instance_count
                }

                instance_dict['segmentation'] = []
                x1 = int(bbox[0])
                y1 = int(bbox[1])-cut_y
                w = int(bbox[2])
                h = int(bbox[3])

                instance_dict['bbox'] = [x1, y1, w, h]
                if image_count >= 0.8 * imgs_num:
                    val_dict['annotations'].append(instance_dict)
                else:
                    train_dict['annotations'].append(instance_dict)


        # image = image / 255.
        # plt.imshow(image)
        # plt.show()
        # break


    json.dump(train_dict, open(root_path+'annotations/'+'instances_train.json', 'w+'))
    json.dump(val_dict, open(root_path+'annotations/'+'instances_val.json', 'w+'))

if __name__ == '__main__':
    to_coco('/home/rvlab/Documents/DRDvideo_processed/')
