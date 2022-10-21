from PIL import Image

import random
import json
import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from shapely.strtree import STRtree
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import pandas as pd
from shapely.geometry import Point, MultiPolygon
from shapely.geometry.polygon import Polygon
import xml.etree.ElementTree as et
from numba import jit
import struct
import get_image_size
from itertools import compress

try:
    import pyspng
except ImportError:
    pyspng = None

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

###################################
######### CAMELYON BASE ###########
###################################

class camelyonBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self.test_label_csv = pd.read_csv('/data/public/public_access/CAMELYON16/test_reference.csv', header=None)
        self.train_anno_xml_dir = '/data/public/public_access/CAMELYON16/training/lesion_annotations'
        self.test_anno_xml_dir = '/data/public/public_access/CAMELYON16/testing/lesion_annotations'
        self.anno_mode = True # whether to use annotations to sample cancer patches
        self.min_count = 64
        self._prepare()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.rgb_data[i], self.case_labels[i]

    def _parse_xml(self, xml_path):
        '''
        Function to parse coordinates in XML format into a list of tuples
        Arguments:
           - xml_filename: XML filename
           - xml_directory: directory where the XML is located

        Returns:
           - coordinates: a list of tuples containing coordinates, divided
                          into segments ([[segment_1], [segment_2], etc])
                          where each segment consists of (x, y) tuples
        '''
        with open(xml_path, 'rt') as f:
            tree = et.parse(f)
            root = tree.getroot()

        # Get the number of coordinates lists inside the xml file
        count = 0
        for Annotation in root.findall('.//Annotation'):
            count += 1

        # Make a list of tuples containing the coordinates
        temp = []
        for Coordinate in root.findall('.//Coordinate'):
            order = float(Coordinate.get('Order'))
            x = float(Coordinate.get('X'))
            y = float(Coordinate.get('Y'))
            temp.append((order, x,y))

        # Separate list of tuples into lists depending on how many segments are annotated
        coordinates = [[] for i in range(count)]
        i = -1
        for j in range(len(temp)):
            if temp[j][0] == 0:
                i += 1
            x = temp[j][1]
            y = temp[j][2]
            coordinates[i].append((x,y))

        return coordinates

    def _prepare(self):
        raise NotImplementedError()

    def _load(self):

        with open(self.rgb_txt_filelist) as json_file:
            self.relpaths = json.load(json_file)

        self.rgb_relpaths = []
        self.rgb_abspaths = []
        self.case_ids = []
        self.case_labels = []

        norm_raw_rgb_relpaths = []
        norm_raw_rgb_abspaths = []
        norm_raw_case_ids = []
        norm_raw_case_labels = []

        canc_raw_rgb_relpaths = []
        canc_raw_rgb_abspaths = []
        canc_raw_case_ids = []
        canc_raw_case_labels = []

        sample_ids = list(self.relpaths.keys())

        tot_canc = 0
        tot_norm_case = 0

        for id in sample_ids:
            if self.anno_mode:
                sample = list(compress(self.relpaths[id][str(self.base_res)][0], self.relpaths[id][str(self.base_res)][1]))
            else:
                sample = self.relpaths[id][str(self.base_res)][0]
            if 'normal' in id:
                TRAIN_FLAG = True
                tot_norm_case += 1
            elif 'tumor' in id:
                TRAIN_FLAG = True
                tot_canc += len(sample)
            else:
                TRAIN_FLAG = False
                if self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal':
                    tot_norm_case += 1
                else:
                    tot_canc += len(sample)

        norm_factor = tot_canc // tot_norm_case
        if (norm_factor % self.min_count) != 0:
            norm_factor += (self.min_count - (norm_factor % self.min_count))

        for id in sample_ids:
            if self.anno_mode:
                sample = list(compress(self.relpaths[id][str(self.base_res)][0], self.relpaths[id][str(self.base_res)][1]))
                sample_meta = self.relpaths[id][str(self.base_res)][0]
                random.shuffle(sample)
                random.shuffle(sample_meta)

                if 'normal' in id:
                    if len(sample) < norm_factor:
                        sample += random.sample(sample_meta, norm_factor - len(sample))
                elif 'tumor' in id:
                    if (len(sample) < self.min_count):
                        sample += random.sample(sample_meta, self.min_count - len(sample))
                else:
                    if (self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal'):
                        if len(sample) < norm_factor:
                            sample += random.sample(sample_meta, norm_factor - len(sample))
                    else:
                        if (len(sample) < self.min_count):
                            sample += random.sample(sample_meta, k=self.min_count - len(sample))

            else:
                sample = self.relpaths[id][str(self.base_res)][0]
                random.shuffle(sample)

                if 'normal' in id:
                    if len(sample) < norm_factor:
                        sample += random.sample(sample, norm_factor - len(sample))
                elif 'tumor' in id:
                    if (len(sample) < self.min_count):
                        sample += random.choices(sample, k=self.min_count - len(sample))
                else:
                    if (self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal'):
                        if len(sample) < norm_factor:
                            sample += random.sample(sample, norm_factor - len(sample))
                    else:
                        if (len(sample) < self.min_count):
                            sample += random.choices(sample, k=self.min_count - len(sample))

            if (len(sample) % self.min_count) != 0:
                sample += random.sample(sample, self.min_count - (len(sample) % self.min_count))

            if self.SR_mode:
                if 'normal' in id:
                    norm_raw_case_labels += [0] * norm_factor
                    norm_raw_case_ids += [id] * norm_factor
                    norm_raw_rgb_relpaths += sample[:norm_factor]
                    norm_raw_rgb_abspaths += [os.path.join(self.rgb_root, p) for p in sample[:norm_factor]]
                elif 'tumor' in id:
                    canc_raw_case_labels += [1] * len(sample)
                    canc_raw_case_ids += [id] * len(sample)
                    canc_raw_rgb_relpaths += sample
                    canc_raw_rgb_abspaths += [os.path.join(self.rgb_root, p) for p in sample]
                else:
                    if self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal':
                        norm_raw_case_labels += [0] * norm_factor
                        norm_raw_case_ids += [id] * norm_factor
                        norm_raw_rgb_relpaths += sample[:norm_factor]
                        norm_raw_rgb_abspaths += [os.path.join(self.rgb_root, p) for p in sample[:norm_factor]]
                    else:
                        canc_raw_case_labels += [1] * len(sample)
                        canc_raw_case_ids += [id] * len(sample)
                        canc_raw_rgb_relpaths += sample
                        canc_raw_rgb_abspaths += [os.path.join(self.rgb_root, p) for p in sample]
            else:
                if 'normal' in id:
                    sample = random.sample(sample, norm_factor)
                elif 'tumor' in id:
                    sample = sample
                else:
                    if (self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal'):
                        sample = random.sample(sample, norm_factor)
                for i in range(0, len(sample), self.min_count):
                    sub_sample = sample[i:i + self.min_count]
                    if 'normal' in id:
                        norm_raw_case_labels.append(0)
                        norm_raw_case_ids.append(id)
                        norm_raw_rgb_relpaths.append(sub_sample)
                        norm_raw_rgb_abspaths.append(
                            [os.path.join(self.rgb_root, p) for p in sub_sample])
                    elif 'tumor' in id:
                        canc_raw_case_labels.append(1)
                        canc_raw_case_ids.append(id)
                        canc_raw_rgb_relpaths.append(sub_sample)
                        canc_raw_rgb_abspaths.append(
                            [os.path.join(self.rgb_root, p) for p in sub_sample])
                    else:
                        if self.test_label_csv[self.test_label_csv[0] == id].values[0][1] == 'Normal':
                            norm_raw_case_labels.append(0)
                            norm_raw_case_ids.append(id)
                            norm_raw_rgb_relpaths.append(sub_sample)
                            norm_raw_rgb_abspaths.append(
                                [os.path.join(self.rgb_root, p) for p in sub_sample])
                        else:
                            canc_raw_case_labels.append(1)
                            canc_raw_case_ids.append(id)
                            canc_raw_rgb_relpaths.append(sub_sample)
                            canc_raw_rgb_abspaths.append(
                                [os.path.join(self.rgb_root, p) for p in sub_sample])
        if TRAIN_FLAG:
            pr_str = 'Training'
        else:
            pr_str = 'Eval.'

        if self.SR_mode:
            print('{} Instance CAMELYON dataloader initialized! Normal ins. count: {} Cancer ins. count: {}'.format(pr_str, len(norm_raw_case_ids), len(canc_raw_case_ids)))
        else:
            print('{} Set CAMELYON dataloader initialized with {} instances per set! Normal set count: {} Cancer set count: {}'.format(pr_str, self.min_count,
                len(norm_raw_case_ids), len(canc_raw_case_ids)))

        self.case_ids = (canc_raw_case_ids + norm_raw_case_ids)
        self.rgb_relpaths = (canc_raw_rgb_relpaths + norm_raw_rgb_relpaths)
        self.rgb_abspaths = (canc_raw_rgb_abspaths + norm_raw_rgb_abspaths)
        self.case_labels = (canc_raw_case_labels + norm_raw_case_labels)

        # add conditional labels
        labels = {
            "case_ids": np.array(self.case_ids),
            "relpath": np.array(self.rgb_relpaths),
            "class_label": np.array(self.case_labels)
        }

        self.rgb_data = self.rgb_abspaths

class camelyonTrain(camelyonBase):
    NAME = "camelyon_train"

    def __init__(self, process_images=True, data_root=None, SR_mode=False, base_res=256, max_per_case=9999999, **kwargs):
        self.base_res = base_res
        self.max_per_case = max_per_case
        self.SR_mode = SR_mode
        self.process_images = process_images
        self.rgb_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        @jit(nopython=True)
        def ray_tracing(x, y, poly):
            n = len(poly)
            inside = False
            p2x = 0.0
            p2y = 0.0
            xints = 0.0
            p1x, p1y = poly[0]
            for i in range(n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        self.random_crop = retrieve(self.config, "/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root, "rgb_filelist_train.json")
        if not tdu.is_prepared(self.rgb_root + "/tdu_train"):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root))

            # make the dict for our cases
            case_patch_dict = {}

            for subdir in os.listdir(self.rgb_root):
                if ('normal_' in subdir) or ('tumor_' in subdir):
                    # we need to filter and store the data
                    case_patch_dict[subdir] = {256: [[], []], 512: [[], []], 1024: [[], []], 2048: [[], []], 4096: [[], []]}
                    for dir_res in os.listdir(os.path.join(self.rgb_root, subdir)):
                        print('Appending {}...'.format(self.rgb_root + '/' + subdir + '/' + dir_res))
                        res = int(dir_res.split(subdir)[-1][1:])
                        split_coord_delta_x = list(range(0, 4096, res)) * int(4096 / res)
                        split_coord_delta_y = np.repeat(list(range(0, 4096, res)),
                                                        int(4096 / res)).tolist()
                        coord_delts = list(zip(split_coord_delta_x, split_coord_delta_y))
                        temp_list = [os.path.relpath(p, start=self.rgb_root) for p in
                                    glob.glob(os.path.join((self.rgb_root + '/' +  subdir + '/' + dir_res), '*.png'))]

                        if len(temp_list) == 0:
                            continue

                        # image size filtering
                        clean_res = []
                        for ins in temp_list:
                            crop_size_w, crop_size_h = get_image_size.get_image_size(
                                os.path.join(self.rgb_root, ins))
                            if (crop_size_w != res) or (crop_size_h != res):
                                continue
                            else:
                                clean_res.append(ins)

                        print('Removed {} samples by resolution filtering in case {} for resolution {}...'.format(
                            len(temp_list) - len(clean_res), subdir, res))

                        # tumor label addition
                        t_label = [True] * len(clean_res)
                        try:
                            if 'normal' not in subdir:
                                if 'tumor' in subdir:
                                    anno_data = self._parse_xml(self.train_anno_xml_dir + '/{}.xml'.format(subdir))
                                    # get valid patches
                                    polygons_bbox = [Polygon(canc_anno) for canc_anno in anno_data]
                                    polygons = [np.array(canc_anno) for canc_anno in anno_data]
                                    bboxes = [polygon.bounds for polygon in polygons_bbox]
                                    for idx, crop_id in enumerate(clean_res):
                                        delt_val = int(crop_id.split('subpatch_')[-1].split('.')[0])
                                        x = int(crop_id.split('x_')[-1].split('_y_')[0]) + coord_delts[delt_val][0]
                                        y = int(crop_id.split('_y_')[-1].split('_subpatch')[0]) + coord_delts[delt_val][1]
                                        raw_points = [(x, y), (x + res, y), (x, y + res),
                                                      (x + res, y + res)]
                                        if any(((bbox[0] <= point[0]) and (point[0] <= bbox[2])
                                                and (bbox[1] <= point[1]) and (point[1] <= bbox[3]))
                                               for point in raw_points for bbox in bboxes):
                                            if any(ray_tracing(point[0], point[1], polygon) for point in raw_points for
                                                   polygon in polygons):
                                                continue
                                            else:
                                                t_label[idx] = False
                                        else:
                                            t_label[idx] = False

                                    print('Adjusted {} at {} resolution yielding a total of {} valid patches...'.format(subdir, res, sum(t_label)))
                            else:
                                print('Normal Case {}, no annotations at resolution {}...'.format(subdir, res))
                        except:
                            print('No available annotations for {} at resolution {}...'.format(subdir, res))

                        print('Case {} at resolution {} has been processed, total {} crops.'.format(subdir, res, len(clean_res)))
                        print('**' * 40)
                        # append the data
                        case_patch_dict[subdir][res][0] = clean_res
                        case_patch_dict[subdir][res][1] = t_label

            with open(self.rgb_txt_filelist, "w") as f:
                json.dump(case_patch_dict, f)

            tdu.mark_prepared(self.rgb_root + "/tdu_train")


class camelyonValidation(camelyonBase):
    NAME = "camelyon_validation"

    def __init__(self, process_images=True, data_root=None, SR_mode=False, base_res=256, max_per_case=9999999,  **kwargs):
        self.max_per_case = max_per_case
        self.base_res = base_res
        self.SR_mode = SR_mode
        self.rgb_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        @jit(nopython=True)
        def ray_tracing(x, y, poly):
            n = len(poly)
            inside = False
            p2x = 0.0
            p2y = 0.0
            xints = 0.0
            p1x, p1y = poly[0]
            for i in range(n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        self.random_crop = retrieve(self.config, "/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root, "rgb_filelist_val.json")

        if not tdu.is_prepared(self.rgb_root + "/tdu_val"):
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root))

            # make the dict for our cases
            case_patch_dict = {}

            for subdir in os.listdir(self.rgb_root):
                if ('test_' in subdir):
                    # we need to filter and store the data
                    case_patch_dict[subdir] = {256: [[], []], 512: [[], []], 1024: [[], []], 2048: [[], []],
                                               4096: [[], []]}
                    for dir_res in os.listdir(os.path.join(self.rgb_root, subdir)):
                        print('Appending {}...'.format(self.rgb_root + '/' + subdir + '/' + dir_res))
                        res = int(dir_res.split(subdir)[-1][1:])
                        split_coord_delta_x = list(range(0, 4096, res)) * int(4096 / res)
                        split_coord_delta_y = np.repeat(list(range(0, 4096, res)),
                                                        int(4096 / res)).tolist()
                        coord_delts = list(zip(split_coord_delta_x, split_coord_delta_y))
                        temp_list = [os.path.relpath(p, start=self.rgb_root) for p in
                                     glob.glob(os.path.join((self.rgb_root + '/' + subdir + '/' + dir_res), '*.png'))]

                        if len(temp_list) == 0:
                            continue

                        # image size filtering
                        clean_res = []
                        for ins in temp_list:
                            crop_size_w, crop_size_h = get_image_size.get_image_size(
                                os.path.join(self.rgb_root, ins))
                            if (crop_size_w != res) or (crop_size_h != res):
                                continue
                            else:
                                clean_res.append(ins)

                        print('Removed {} samples by resolution filtering in case {} for resolution {}...'.format(
                            len(temp_list) - len(clean_res), subdir, res))

                        # tumor label addition
                        t_label = [True] * len(clean_res)
                        try:
                            if self.test_label_csv[self.test_label_csv[0] == subdir].values[0][1] != 'Normal':
                                anno_data = self._parse_xml(self.test_anno_xml_dir + '/{}.xml'.format(subdir))
                                # get valid patches
                                polygons_bbox = [Polygon(canc_anno) for canc_anno in anno_data]
                                polygons = [np.array(canc_anno) for canc_anno in anno_data]
                                bboxes = [polygon.bounds for polygon in polygons_bbox]
                                for idx, crop_id in enumerate(clean_res):
                                    delt_val = int(crop_id.split('subpatch_')[-1].split('.')[0])
                                    x = int(crop_id.split('x_')[-1].split('_y_')[0]) + coord_delts[delt_val][0]
                                    y = int(crop_id.split('_y_')[-1].split('_subpatch')[0]) + coord_delts[delt_val][
                                        1]
                                    raw_points = [(x, y), (x + res, y), (x, y + res),
                                                  (x + res, y + res)]
                                    if any(((bbox[0] <= point[0]) and (point[0] <= bbox[2])
                                            and (bbox[1] <= point[1]) and (point[1] <= bbox[3]))
                                           for point in raw_points for bbox in bboxes):
                                        if any(ray_tracing(point[0], point[1], polygon) for point in raw_points for
                                               polygon in polygons):
                                            continue
                                        else:
                                            t_label[idx] = False
                                    else:
                                        t_label[idx] = False

                                print('Adjusted {} at {} resolution yielding a total of {} valid patches...'.format(
                                    subdir, res, sum(t_label)))
                            else:
                                print('Normal Case {}, no annotations at resolution {}...'.format(subdir, res))
                        except:
                            print('No available annotations for {} at resolution {}...'.format(subdir, res))

                        print('Case {} at resolution {} has been processed, total {} crops.'.format(subdir, res,
                                                                                                    len(clean_res)))
                        print('**' * 40)
                        # append the data
                        case_patch_dict[subdir][res][0] = clean_res
                        case_patch_dict[subdir][res][1] = t_label

            with open(self.rgb_txt_filelist, "w") as f:
                json.dump(case_patch_dict, f)

            tdu.mark_prepared(self.rgb_root + "/tdu_val")


class camelyonTest(camelyonBase):
    NAME = "camelyon_test"

    def __init__(self, process_images=True, data_root=None, SR_mode=False, base_res=256, max_per_case=9999999, **kwargs):
        self.max_per_case = max_per_case
        self.base_res = base_res
        self.SR_mode = SR_mode
        self.rgb_root = data_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        @jit(nopython=True)
        def ray_tracing(x, y, poly):
            n = len(poly)
            inside = False
            p2x = 0.0
            p2y = 0.0
            xints = 0.0
            p1x, p1y = poly[0]
            for i in range(n + 1):
                p2x, p2y = poly[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xints:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        self.random_crop = retrieve(self.config, "/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root, "rgb_filelist_test.json")
        if not tdu.is_prepared(self.rgb_root + "/tdu_test"):
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root))

            # make the dict for our cases
            case_patch_dict = {}

            for subdir in os.listdir(self.rgb_root):
                if ('test_' in subdir):
                    # we need to filter and store the data
                    case_patch_dict[subdir] = {256: [[], []], 512: [[], []], 1024: [[], []], 2048: [[], []],
                                               4096: [[], []]}
                    for dir_res in os.listdir(os.path.join(self.rgb_root, subdir)):
                        print('Appending {}...'.format(self.rgb_root + '/' + subdir + '/' + dir_res))
                        res = int(dir_res.split(subdir)[-1][1:])
                        split_coord_delta_x = list(range(0, 4096, res)) * int(4096 / res)
                        split_coord_delta_y = np.repeat(list(range(0, 4096, res)),
                                                        int(4096 / res)).tolist()
                        coord_delts = list(zip(split_coord_delta_x, split_coord_delta_y))
                        temp_list = [os.path.relpath(p, start=self.rgb_root) for p in
                                     glob.glob(os.path.join((self.rgb_root + '/' + subdir + '/' + dir_res), '*.png'))]

                        if len(temp_list) == 0:
                            continue

                        # image size filtering
                        clean_res = []
                        for ins in temp_list:
                            crop_size_w, crop_size_h = get_image_size.get_image_size(
                                os.path.join(self.rgb_root, ins))
                            if (crop_size_w != res) or (crop_size_h != res):
                                continue
                            else:
                                clean_res.append(ins)

                        print('Removed {} samples by resolution filtering in case {} for resolution {}...'.format(
                            len(temp_list) - len(clean_res), subdir, res))

                        # tumor label addition
                        t_label = [True] * len(clean_res)
                        try:
                            if self.test_label_csv[self.test_label_csv[0] == subdir].values[0][1] != 'Normal':
                                anno_data = self._parse_xml(self.test_anno_xml_dir + '/{}.xml'.format(subdir))
                                # get valid patches
                                polygons_bbox = [Polygon(canc_anno) for canc_anno in anno_data]
                                polygons = [np.array(canc_anno) for canc_anno in anno_data]
                                bboxes = [polygon.bounds for polygon in polygons_bbox]
                                for idx, crop_id in enumerate(clean_res):
                                    delt_val = int(crop_id.split('subpatch_')[-1].split('.')[0])
                                    x = int(crop_id.split('x_')[-1].split('_y_')[0]) + coord_delts[delt_val][0]
                                    y = int(crop_id.split('_y_')[-1].split('_subpatch')[0]) + coord_delts[delt_val][
                                        1]
                                    raw_points = [(x, y), (x + res, y), (x, y + res),
                                                  (x + res, y + res)]
                                    if any(((bbox[0] <= point[0]) and (point[0] <= bbox[2])
                                            and (bbox[1] <= point[1]) and (point[1] <= bbox[3]))
                                           for point in raw_points for bbox in bboxes):
                                        if any(ray_tracing(point[0], point[1], polygon) for point in raw_points for
                                               polygon in polygons):
                                            continue
                                        else:
                                            t_label[idx] = False
                                    else:
                                        t_label[idx] = False

                                print('Adjusted {} at {} resolution yielding a total of {} valid patches...'.format(
                                    subdir, res, sum(t_label)))
                            else:
                                print('Normal Case {}, no annotations at resolution {}...'.format(subdir, res))
                        except:
                            print('No available annotations for {} at resolution {}...'.format(subdir, res))

                        print('Case {} at resolution {} has been processed, total {} crops.'.format(subdir, res,
                                                                                                    len(clean_res)))
                        print('**' * 40)
                        # append the data
                        case_patch_dict[subdir][res][0] = clean_res
                        case_patch_dict[subdir][res][1] = t_label

            with open(self.rgb_txt_filelist, "w") as f:
                json.dump(case_patch_dict, f)

            tdu.mark_prepared(self.rgb_root + "/tdu_test")

###################################
######### CAMELYON SR #############
###################################


class camelyonSR(Dataset):
    def __init__(self, size_h=None, size_w=None, mask_mode='sem',
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, precision=32, base_res=256, max_per_case=2000):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.SR_mode = True
        self.mask_mode = mask_mode
        self.precision = precision
        self.base_res = base_res
        self.max_per_case = max_per_case
        self.base = self.get_base()
        assert size_h
        assert size_w
        self.size_h = size_h
        self.size_w = size_w
        self.LR_size = [int(size_h / downscale_f), int(size_w / downscale_f)]
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
        }[degradation]

        self.rgb_image_rescaler = albumentations.Resize(size_h, size_w, interpolation=interpolation_fn)
        self.mask_rescaler = albumentations.Resize(size_h, size_w, interpolation=interpolation_fn)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        self.pil_interpolation = degradation.startswith("pil_")

        if self.pil_interpolation:
            self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

        else:
            self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                      interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example_rgb, example_label = self.base[i]
        with self._open_file(example_rgb) as f:
            if pyspng is not None:
                rgb_image = pyspng.load(f.read())
            else:
                rgb_image = Image.open(f)
                rgb_image = np.array(rgb_image).astype(np.uint8)

        h_len = rgb_image.shape[0]
        w_len = rgb_image.shape[1]

        crop_h_len = h_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_h_len = int(crop_h_len)

        crop_w_len = w_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_w_len = int(crop_w_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_h_len, width=crop_w_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_h_len, width=crop_w_len)

        image = self.cropper(image=rgb_image)["image"]
        image = self.rgb_image_rescaler(image=image)["image"]


        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example = {}
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)
        example["class_label"] = example_label

        return example

    def _open_file(self, fname):
        return open(fname, 'rb')


class camelyonSRTrain(camelyonSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTrain(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            # SR_mode=False,
                            SR_mode=True,
                            base_res=self.base_res,
                            max_per_case=self.max_per_case)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonSRValidation(camelyonSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonValidation(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=True,
                            # SR_mode=False,
                            base_res=self.base_res,
                            max_per_case=self.max_per_case)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonSRTest(camelyonSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTest(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=True,
                            # SR_mode=False,
                            base_res=self.base_res,
                            max_per_case=self.max_per_case)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)

###################################
######## CAMELYON SET #############
###################################


class camelyonSet(Dataset):
    def __init__(self, size_h=None, size_w=None, base_res=256, interpolation="bicubic", flip_p=0.5, set_size=20):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.SR_mode = False
        self.flip_p = flip_p
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.set_size = set_size
        self.flip_h = transforms.RandomHorizontalFlip(p=flip_p)
        self.flip_v = transforms.RandomVerticalFlip(p=flip_p)
        self.base_res = base_res
        self.base = self.get_base()
        assert size_h
        assert size_w
        self.size_h = size_h
        self.size_w = size_w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = {}
        example["image"] = np.zeros((self.set_size, self.size_h, self.size_w, 3)).astype(np.float32) # currently supports rgb only
        raw_set, example_label = self.base[i]
        set_rgb = random.sample(raw_set, self.set_size)

        for id, instance in enumerate(set_rgb):
            with self._open_file(instance) as f:
                if pyspng is not None:
                    rgb_image = pyspng.load(f.read())
                    if random.random() < self.flip_p:
                        rgb_image = rgb_image[::-1, :, :]
                    if random.random() < self.flip_p:
                        rgb_image = rgb_image[:, ::-1, :]
                else:
                    rgb_image = Image.open(f)
                    rgb_image = self.flip_h(rgb_image)
                    rgb_image = self.flip_v(rgb_image)
                    rgb_image = np.array(rgb_image).astype(np.uint8)

            if (self.base_res != self.size_h) or (self.base_res != self.size_w): # slow, we usually dont resize camelyon crops here
                rgb_image = Image.fromarray(rgb_image)
                rgb_image = rgb_image.resize((self.size_h, self.size_w), resample=self.interpolation)
                rgb_image = np.array(rgb_image).astype(np.uint8)

            example["image"][id, ...][:] = (rgb_image/127.5 - 1.0).astype(np.float32)[:]
            example["class_label"] = example_label

        return example

    def _open_file(self, fname):
        return open(fname, 'rb')


class camelyonSetTrain(camelyonSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTrain(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=False,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonSetValidation(camelyonSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonValidation(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=False,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonSetTest(camelyonSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTest(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=False,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)

###################################
### CAMELYON INSTANCE #############
###################################


class camelyonInstance(Dataset):
    def __init__(self, size_h=None, size_w=None, base_res=256, interpolation="bicubic", flip_p=0.5, set_size=20):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.SR_mode = False
        self.flip_p = flip_p
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.set_size = set_size
        self.flip_h = transforms.RandomHorizontalFlip(p=flip_p)
        self.flip_v = transforms.RandomVerticalFlip(p=flip_p)
        self.base_res = base_res
        self.base = self.get_base()
        assert size_h
        assert size_w
        self.size_h = size_h
        self.size_w = size_w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = {}
        instance, example_label = self.base[i]

        with self._open_file(instance) as f:
            if pyspng is not None:
                rgb_image = pyspng.load(f.read())
                if random.random() < self.flip_p:
                    rgb_image = rgb_image[::-1, :, :]
                if random.random() < self.flip_p:
                    rgb_image = rgb_image[:, ::-1, :]
            else:
                rgb_image = Image.open(f)
                rgb_image = self.flip_h(rgb_image)
                rgb_image = self.flip_v(rgb_image)
                rgb_image = np.array(rgb_image).astype(np.uint8)

        if (self.base_res != self.size_h) or (self.base_res != self.size_w): # slow, we usually dont resize camelyon crops here
            rgb_image = Image.fromarray(rgb_image)
            rgb_image = rgb_image.resize((self.size_h, self.size_w), resample=self.interpolation)
            rgb_image = np.array(rgb_image).astype(np.uint8)

        example["image"] = (rgb_image/127.5 - 1.0).astype(np.float32)
        example["class_label"] = example_label

        return example

    def _open_file(self, fname):
        return open(fname, 'rb')


class camelyonInstanceTrain(camelyonInstance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTrain(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=True,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonInstanceValidation(camelyonInstance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonValidation(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=True,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class camelyonInstanceTest(camelyonInstance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = camelyonTest(process_images=False,
                            data_root='/data/public/public_access/CAMELYON16/CLAM_patches/crops',
                            SR_mode=True,
                            base_res=self.base_res,
                            max_per_case=1000)
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)

