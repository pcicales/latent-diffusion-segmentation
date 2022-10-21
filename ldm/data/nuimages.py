from PIL import Image

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
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

def synset2idx(path_to_yaml="data/index_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

###################################
########## NUIMAGE BASE ###########
###################################

class nuimageBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self.keep_orig_class_label = self.config.get("keep_orig_class_label", False)
        self.process_images = True  # if False we skip loading & processing images and self.data contains filepaths
        self._prepare()
        # self._prepare_synset_to_human()
        # self._prepare_idx_to_synset()
        # self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.rgb_data[i], self.sem_data[i], self.ins_data[i]

    def _prepare(self):
        raise NotImplementedError()

    # def _filter_relpaths(self, relpaths):
    #     ignore = set([
    #         "n06596364_9591.JPEG",
    #     ])
    #     relpaths = [rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
    #     if "sub_indices" in self.config:
    #         indices = str_to_indices(self.config["sub_indices"])
    #         synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
    #         self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
    #         files = []
    #         for rpath in relpaths:
    #             syn = rpath.split("/")[0]
    #             if syn in synsets:
    #                 files.append(rpath)
    #         return files
    #     else:
    #         return relpaths
    #
    # def _prepare_synset_to_human(self):
    #     SIZE = 2655750
    #     URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
    #     self.human_dict = os.path.join(self.root, "synset_human.txt")
    #     if (not os.path.exists(self.human_dict) or
    #             not os.path.getsize(self.human_dict)==SIZE):
    #         download(URL, self.human_dict)
    #
    # def _prepare_idx_to_synset(self):
    #     URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
    #     self.idx2syn = os.path.join(self.root, "index_synset.yaml")
    #     if (not os.path.exists(self.idx2syn)):
    #         download(URL, self.idx2syn)
    #
    # def _prepare_human_to_integer_label(self):
    #     URL = "https://heibox.uni-heidelberg.de/f/2362b797d5be43b883f6/?dl=1"
    #     self.human2integer = os.path.join(self.root, "imagenet1000_clsidx_to_labels.txt")
    #     if (not os.path.exists(self.human2integer)):
    #         download(URL, self.human2integer)
    #     with open(self.human2integer, "r") as f:
    #         lines = f.read().splitlines()
    #         assert len(lines) == 1000
    #         self.human2integer_dict = dict()
    #         for line in lines:
    #             value, key = line.split(":")
    #             self.human2integer_dict[key] = int(value)
    #
    def _load(self):
        with open(self.rgb_txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()

        self.rgb_abspaths = [os.path.join(self.rgb_root, p) for p in self.relpaths]
        self.sem_abspaths = [os.path.join(self.sem_root, p) for p in self.relpaths]
        self.ins_abspaths = [os.path.join(self.ins_root, p) for p in self.relpaths]

        labels = {
            "relpath": np.array(self.relpaths),
        }

        # can maybe remedy this
        # if self.process_images:
        #     self.size = retrieve(self.config, "size", default=256)
        #     self.data = ImagePaths(self.abspaths,
        #                            labels=labels,
        #                            size=self.size,
        #                            random_crop=self.random_crop,
        #                            )
        # else:
        self.rgb_data = self.rgb_abspaths
        self.sem_data = self.sem_abspaths
        self.ins_data = self.ins_abspaths


class nuimageTrain(nuimageBase):
    NAME = "nuimage_train"

    def __init__(self, process_images=True, data_rgb_root=None, data_sem_root=None, data_ins_root=None, **kwargs):
        self.process_images = process_images
        self.rgb_root = data_rgb_root
        self.sem_root = data_sem_root
        self.ins_root = data_ins_root
        super().__init__(**kwargs)

    def _prepare(self):
        self.random_crop = retrieve(self.config, "nuimageTrain/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root[:-4], "rgb_filelist.txt")
        self.sem_txt_filelist = os.path.join(self.rgb_root[:-4], "sem_filelist.txt")
        self.ins_txt_filelist = os.path.join(self.rgb_root[:-4], "ins_filelist.txt")
        if not tdu.is_prepared(self.rgb_root[:-4]):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root[:-4]))

            rgb_datadir = self.rgb_root
            sem_datadir = self.sem_root
            ins_datadir = self.ins_root

            rgb_filelist = glob.glob(os.path.join(rgb_datadir, "*.png"))
            sem_filelist = glob.glob(os.path.join(sem_datadir, "*.png"))
            ins_filelist = glob.glob(os.path.join(ins_datadir, "*.png"))

            rgb_filelist = [os.path.relpath(p, start=rgb_datadir) for p in rgb_filelist]
            sem_filelist = [os.path.relpath(p, start=sem_datadir) for p in sem_filelist]
            ins_filelist = [os.path.relpath(p, start=ins_datadir) for p in ins_filelist]

            rgb_filelist = sorted(rgb_filelist)
            sem_filelist = sorted(sem_filelist)
            ins_filelist = sorted(ins_filelist)

            rgb_filelist = "\n".join(rgb_filelist)+"\n"
            sem_filelist = "\n".join(sem_filelist) + "\n"
            ins_filelist = "\n".join(ins_filelist) + "\n"

            with open(self.rgb_txt_filelist, "w") as f:
                f.write(rgb_filelist)
            with open(self.sem_txt_filelist, "w") as f:
                f.write(sem_filelist)
            with open(self.ins_txt_filelist, "w") as f:
                f.write(ins_filelist)

            tdu.mark_prepared(self.rgb_root[:-4])


class nuimageValidation(nuimageBase):
    NAME = "nuimage_validation"

    def __init__(self, process_images=True, data_rgb_root=None, data_sem_root=None, data_ins_root=None, **kwargs):
        self.rgb_root = data_rgb_root
        self.sem_root = data_sem_root
        self.ins_root = data_ins_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        self.random_crop = retrieve(self.config, "nuimageValidation/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root[:-4], "rgb_filelist.txt")
        self.sem_txt_filelist = os.path.join(self.rgb_root[:-4], "sem_filelist.txt")
        self.ins_txt_filelist = os.path.join(self.rgb_root[:-4], "ins_filelist.txt")
        if not tdu.is_prepared(self.rgb_root[:-4]):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root[:-4]))

            rgb_datadir = self.rgb_root
            sem_datadir = self.sem_root
            ins_datadir = self.ins_root

            rgb_filelist = glob.glob(os.path.join(rgb_datadir, "*.png"))
            sem_filelist = glob.glob(os.path.join(sem_datadir, "*.png"))
            ins_filelist = glob.glob(os.path.join(ins_datadir, "*.png"))

            rgb_filelist = [os.path.relpath(p, start=rgb_datadir) for p in rgb_filelist]
            sem_filelist = [os.path.relpath(p, start=sem_datadir) for p in sem_filelist]
            ins_filelist = [os.path.relpath(p, start=ins_datadir) for p in ins_filelist]

            rgb_filelist = sorted(rgb_filelist)
            sem_filelist = sorted(sem_filelist)
            ins_filelist = sorted(ins_filelist)

            rgb_filelist = "\n".join(rgb_filelist) + "\n"
            sem_filelist = "\n".join(sem_filelist) + "\n"
            ins_filelist = "\n".join(ins_filelist) + "\n"

            with open(self.rgb_txt_filelist, "w") as f:
                f.write(rgb_filelist)
            with open(self.sem_txt_filelist, "w") as f:
                f.write(sem_filelist)
            with open(self.ins_txt_filelist, "w") as f:
                f.write(ins_filelist)

            tdu.mark_prepared(self.rgb_root[:-4])


class nuimageTest(nuimageBase):
    NAME = "nuimage_test"


    def __init__(self, process_images=True, data_rgb_root=None, data_sem_root=None, data_ins_root=None, **kwargs):
        self.rgb_root = data_rgb_root
        self.sem_root = data_sem_root
        self.ins_root = data_ins_root
        self.process_images = process_images
        super().__init__(**kwargs)

    def _prepare(self):
        self.random_crop = retrieve(self.config, "nuimageTest/random_crop",
                                    default=True)
        self.rgb_txt_filelist = os.path.join(self.rgb_root[:-4], "rgb_filelist.txt")
        self.sem_txt_filelist = os.path.join(self.rgb_root[:-4], "sem_filelist.txt")
        self.ins_txt_filelist = os.path.join(self.rgb_root[:-4], "ins_filelist.txt")
        if not tdu.is_prepared(self.rgb_root[:-4]):
            # prep
            print("Preparing dataset {} in {}".format(self.NAME, self.rgb_root[:-4]))

            rgb_datadir = self.rgb_root
            sem_datadir = self.sem_root
            ins_datadir = self.ins_root

            rgb_filelist = glob.glob(os.path.join(rgb_datadir, "*.png"))
            sem_filelist = glob.glob(os.path.join(sem_datadir, "*.png"))
            ins_filelist = glob.glob(os.path.join(ins_datadir, "*.png"))

            rgb_filelist = [os.path.relpath(p, start=rgb_datadir) for p in rgb_filelist]
            sem_filelist = [os.path.relpath(p, start=sem_datadir) for p in sem_filelist]
            ins_filelist = [os.path.relpath(p, start=ins_datadir) for p in ins_filelist]

            rgb_filelist = sorted(rgb_filelist)
            sem_filelist = sorted(sem_filelist)
            ins_filelist = sorted(ins_filelist)

            rgb_filelist = "\n".join(rgb_filelist) + "\n"
            sem_filelist = "\n".join(sem_filelist) + "\n"
            ins_filelist = "\n".join(ins_filelist) + "\n"

            with open(self.rgb_txt_filelist, "w") as f:
                f.write(rgb_filelist)
            with open(self.sem_txt_filelist, "w") as f:
                f.write(sem_filelist)
            with open(self.ins_txt_filelist, "w") as f:
                f.write(ins_filelist)

            tdu.mark_prepared(self.rgb_root[:-4])


###################################
########## NUIMAGE SR #############
###################################


class nuimageSR(Dataset):
    def __init__(self, size_h=None, size_w=None, mask_mode='sem',
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True, precision=32):
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
        self.mask_mode = mask_mode
        self.precision = precision
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
        example_rgb, example_sem, example_ins = self.base[i]
        rgb_image = Image.open(example_rgb)

        rgb_image = np.array(rgb_image).astype(np.uint8)

        h_len = rgb_image.shape[0]
        w_len = rgb_image.shape[1]

        crop_h_len = h_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_h_len = int(crop_h_len)

        crop_w_len = w_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_w_len = int(crop_w_len)

        if self.center_crop:
            if (self.mask_mode == 'sem'):
                sem_image = Image.open(example_sem)
                sem_image = np.array(sem_image).astype(np.uint8)
                self.cropper = albumentations.Compose([albumentations.CenterCrop(height=crop_h_len, width=crop_w_len)],
                                                      additional_targets={'image': 'image', 'sem_mask': 'mask'})
            elif (self.mask_mode == 'ins'):
                sem_image = Image.open(example_sem)
                sem_image = np.array(sem_image).astype(np.uint8)
                ins_image = Image.open(example_ins)
                ins_image = np.array(ins_image).astype(np.uint8)
                self.cropper = albumentations.Compose([albumentations.CenterCrop(height=crop_h_len, width=crop_w_len)],
                                                      additional_targets={'image': 'image', 'sem_mask': 'mask', 'ins_mask': 'mask'})
            else:
                self.cropper = albumentations.CenterCrop(height=crop_h_len, width=crop_w_len)

        else:
            if (self.mask_mode == 'sem'):
                sem_image = Image.open(example_sem)
                sem_image = np.array(sem_image).astype(np.uint8)
                self.cropper = albumentations.Compose([albumentations.RandomCrop(height=crop_h_len, width=crop_w_len)],
                                                      additional_targets={'image': 'image', 'sem_mask': 'mask'})
            elif (self.mask_mode == 'ins'):
                sem_image = Image.open(example_sem)
                sem_image = np.array(sem_image).astype(np.uint8)
                ins_image = Image.open(example_ins)
                ins_image = np.array(ins_image).astype(np.uint8)
                self.cropper = albumentations.Compose([albumentations.RandomCrop(height=crop_h_len, width=crop_w_len)],
                                                      additional_targets={'image': 'image', 'sem_mask': 'mask',
                                                                          'ins_mask': 'mask'})
            else:
                self.cropper = albumentations.RandomCrop(height=crop_h_len, width=crop_w_len)


        if (self.mask_mode == 'sem'):
            crop_xform = self.cropper(image=rgb_image, sem_mask=sem_image)
            rgb_image = crop_xform["image"]
            sem_image = crop_xform["sem_mask"]

            rgb_image = self.rgb_image_rescaler(image=rgb_image)["image"]
            sem_image = self.mask_rescaler(image=sem_image)["image"]

            image = np.concatenate((rgb_image, sem_image[:, :, np.newaxis]), axis=2)

        elif (self.mask_mode == 'ins'):
            crop_xform = self.cropper(image=rgb_image, sem_mask=sem_image, ins_mask=ins_image)
            rgb_image = crop_xform["image"]
            sem_image = crop_xform["sem_mask"]
            ins_image = crop_xform["ins_mask"]

            rgb_image = self.rgb_image_rescaler(image=rgb_image)["image"]
            sem_image = self.mask_rescaler(image=sem_image)["image"]
            ins_image = self.mask_rescaler(image=ins_image)["image"]

            image = np.concatenate((rgb_image, sem_image[:, :, np.newaxis], ins_image[:, :, np.newaxis]), axis=2)

        else:
            image = self.cropper(image=rgb_image)["image"]
            image = self.rgb_image_rescaler(image=image)["image"]


        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        else:
            LR_image = self.degradation_process(image=image)["image"]

        example = {}
        # if self.precision == 16:
        #     example["image"] = (image/127.5 - 1.0).astype(np.float16)
        #     example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float16)
        # else:
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        return example


class nuimageSRTrain(nuimageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = nuimageTrain(process_images=False,
                            data_rgb_root='/data/public/public_access/nuimages/RGBA/train/RGB',
                            data_sem_root='/data/public/public_access/nuimages/RGBA/train/sem',
                            data_ins_root='/data/public/public_access/nuimages/RGBA/train/ins')
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class nuimageSRValidation(nuimageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = nuimageValidation(process_images=False,
                            data_rgb_root='/data/public/public_access/nuimages/RGBA/val/RGB',
                            data_sem_root='/data/public/public_access/nuimages/RGBA/val/sem',
                            data_ins_root='/data/public/public_access/nuimages/RGBA/val/ins')
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)


class nuimageSRTest(nuimageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        dset = nuimageTest(process_images=False,
                            data_rgb_root='/data/public/public_access/nuimages/RGBA/test/RGB',
                            data_sem_root='/data/public/public_access/nuimages/RGBA/test/sem',
                            data_ins_root='/data/public/public_access/nuimages/RGBA/test/ins')
        indices = list(range(len(dset.rgb_data)))
        return Subset(dset, indices)

