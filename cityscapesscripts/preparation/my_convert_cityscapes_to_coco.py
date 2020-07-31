#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

# This file is copy from https://github.com/facebookresearch/Detectron/tree/master/tools

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys

import cityscapesscripts.evaluation.instances2dict_with_polygons as cs


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="cocostuff, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]

    return box_from_poly


def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box


def convert_coco_stuff_mat(data_dir, out_dir):
    """Convert to png and save json with path. This currently only contains
    the segmentation labels for objects+stuff in cocostuff - if we need to
    combine with other labels from original COCO that will be a TODO."""
    sets = ['train', 'val']
    categories = []
    json_name = 'coco_stuff_%s.json'
    ann_dict = {}
    for data_set in sets:
        file_list = os.path.join(data_dir, '%s.txt')
        images = []
        with open(file_list % data_set) as f:
            for img_id, img_name in enumerate(f):
                img_name = img_name.replace('coco', 'COCO').strip('\n')
                image = {}
                mat_file = os.path.join(
                    data_dir, 'annotations/%s.mat' % img_name)
                data = h5py.File(mat_file, 'r')
                labelMap = data.get('S')
                if len(categories) == 0:
                    labelNames = data.get('names')
                    for idx, n in enumerate(labelNames):
                        categories.append(
                            {"id": idx, "name": ''.join(chr(i) for i in data[
                                n[0]])})
                    ann_dict['categories'] = categories
                scipy.misc.imsave(
                    os.path.join(data_dir, img_name + '.png'), labelMap)
                image['width'] = labelMap.shape[0]
                image['height'] = labelMap.shape[1]
                image['file_name'] = img_name
                image['seg_file_name'] = img_name
                image['id'] = img_id
                images.append(image)
        ann_dict['images'] = images
        print("Num images: %s" % len(images))
        with open(os.path.join(out_dir, json_name % data_set), 'wb') as outfile:
            outfile.write(json.dumps(ann_dict))


# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)


def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'gtFine_val',
        'gtFine_train',
        'gtFine_test',

        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'gtFine\\val',
        'gtFine\\train',
        'gtFine\\test',

        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    # category_dict = {}

    # category_no_need = [
    #     'static',
    #     'ego vehicle',
    #     'rectification border',
    #     'out of roi',
    #     'dynamic',
    #     'terrain'
    # ]

    category_need = [{'id': 1, 'name': 'sky'}, {'id': 2, 'name': 'road'}, {'id': 3, 'name': 'sidewalk'},
     {'id': 4, 'name': 'vegetation'}, {'id': 5, 'name': 'pole'}, {'id': 6, 'name': 'building'},
     {'id': 7, 'name': 'traffic sign'}, {'id': 8, 'name': 'fence'}, {'id': 9, 'name': 'person'},
     {'id': 10, 'name': 'car'}, {'id': 11, 'name': 'cargroup'}, {'id': 12, 'name': 'bicycle'},
     {'id': 13, 'name': 'rider'}, {'id': 14, 'name': 'parking'}, {'id': 15, 'name': 'license plate'},
     {'id': 16, 'name': 'traffic light'}, {'id': 17, 'name': 'truck'}, {'id': 18, 'name': 'motorcycle'},
     {'id': 19, 'name': 'train'}, {'id': 20, 'name': 'bus'}, {'id': 21, 'name': 'rail track'},
     {'id': 22, 'name': 'ground'}, {'id': 23, 'name': 'wall'}, {'id': 24, 'name': 'polegroup'},
     {'id': 25, 'name': 'bicyclegroup'}, {'id': 26, 'name': 'persongroup'}, {'id': 27, 'name': 'ridergroup'},
     {'id': 28, 'name': 'bridge'}, {'id': 29, 'name': 'trailer'}, {'id': 30, 'name': 'caravan'},
     {'id': 31, 'name': 'guard rail'}, {'id': 32, 'name': 'tunnel'}, {'id': 33, 'name': 'motorcyclegroup'},
     {'id': 34, 'name': 'trunkgroup'}]

    category_id_name_dct = dict([(itm['id'],itm['name']) for itm in category_need])
    category_name_id_dct = dict([(itm['name'],itm['id']) for itm in category_need])

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)

        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in % data_set.split('_')[0]):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth']
                    image['height'] = json_ann['imgHeight']
                    image['file_name'] = filename[:-len(
                        ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
                    # image['seg_file_name'] = filename[:-len(
                    #     ends_in % data_set.split('_')[0])] + \
                    #                          '%s_instanceIds.png' % data_set.split('_')[0]
                    images.append(image)

                    # fullname = os.path.join(root, image['seg_file_name'])
                    # objects = cs.instances2dict_with_polygons(
                    #     [fullname], verbose=False)[fullname]

                    for obj in json_ann['objects']:
                        # if object_cls not in category_instancesonly:
                        #     continue  # skip non-instance categories
                            if obj['label'] not in category_name_id_dct.keys():
                                print('Warning: no need category.')
                                continue

                            if obj['polygon'] == []:
                                print('Warning: empty polygon.')
                                continue  # skip non-instance categories

                            # if len(obj['polygon']) <= 2:
                            #     print('Warning: invalid polygon.')
                            #     continue  # skip non-instance categories

                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = obj['polygon']

                            ann['category_id'] = category_name_id_dct[obj['label']]
                            ann['iscrowd'] = 0
                            ann['area'] = 0

                            xyxy_box = poly_to_box(ann['segmentation'])
                            xywh_box = xyxy_to_xywh(xyxy_box)
                            ann['bbox'] = xywh_box

                            annotations.append(ann)

        ann_dict['images'] = images
        categories = [{"id": category_name_id_dct[name], "name": name} for name in
                      category_name_id_dct]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cityscapes":
        convert_cityscapes_instance_only(args.datadir, args.outdir)
    elif args.dataset == "cocostuff":
        convert_coco_stuff_mat(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)