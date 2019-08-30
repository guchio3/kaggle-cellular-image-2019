import gc
import os
from functools import partial
from itertools import chain
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
# import rxrx.io as rio
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils.logs import sel_log

# sys.path.append('../tools/utils/rxrx1-utils')

IMAGE_SIZE = 512


def _load_imgs_from_ids(id_pair, mode):
    _id, label = id_pair
    images = []
    split_id = _id.split('_')
    filename_base = f'./mnt/inputs/{mode}/{split_id[0]}/' \
                    f'Plate{split_id[1]}/{split_id[2]}'
    for site in [1, 2]:
        _images = []
        for w in [1, 2, 3, 4, 5, 6]:
            # 0 means gray scale
            _images.append(
                cv2.imread(f'{filename_base}_s{site}_w{w}.png', 0))
        images.append(
            np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6))
    return images


class CellularImageDataset(Dataset):
    def __init__(self, mode, ids, augment,
                 visualize=False, logger=None):
        '''
        ids : id_code
        '''
        self.mode = mode
        self.visualize = visualize
        self.logger = logger

        if mode == "test":
            labels = [0] * len(ids)
        else:  # train or valid
            labels = pd.read_csv('./mnt/inputs/origin/train.csv.zip')\
                .set_index('id_code').loc[ids]['sirna'].values
        self.images, self.labels = self._parse_ids(mode, ids, labels)

        # load validation
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        # img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        assert img.shape == (6, IMAGE_SIZE, IMAGE_SIZE)

        return (torch.tensor(img), torch.tensor(self.labels[idx]))

    def _parse_ids(self, mode, ids, original_labels):
        #        ids = ids[:300]
        #        original_labels = original_labels[:300]
        assert len(ids) == len(original_labels)
        images = []
        # 2 mean sites
        labels = list(chain.from_iterable(
            [[label] * 2 for label in original_labels]))
        sel_log('now loading images ...', self.logger)
        with Pool(os.cpu_count()) as p:
            # self だとエラー
            iter_func = partial(_load_imgs_from_ids, mode=mode)
            imap = p.imap_unordered(iter_func, list(zip(ids, original_labels)))
            images = list(tqdm(imap, total=len(ids)))
            p.close()
            p.join()
            gc.collect()
        images = list(chain.from_iterable(images))
        # images = np.array(images)

        return images, labels


class CellularImageDatasetV2(Dataset):
    def __init__(self, mode, ids, augment,
                 visualize=False, logger=None):
        '''
        ids : id_code
        '''
        self.mode = mode
        self.visualize = visualize
        self.logger = logger

        if mode == "test":
            labels = [0] * len(ids)
        else:  # train or valid
            labels = pd.read_csv('./mnt/inputs/origin/train.csv.zip')\
                .set_index('id_code').loc[ids]['sirna'].values
        self.image_files, self.labels = self._parse_ids(mode, ids, labels)

        # load validation
        assert len(self.image_files) == len(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_files = self.image_files[idx]
        img = self._load_one_img(img_files)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        # img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        assert img.shape == (6, IMAGE_SIZE, IMAGE_SIZE)

        return (torch.tensor(img), torch.tensor(self.labels[idx]))

    def _parse_ids(self, mode, ids, original_labels):
        # ids = ids[:300]
        # original_labels = original_labels[:300]
        assert len(ids) == len(original_labels)
        # 2 mean sites
        labels = list(chain.from_iterable(
            [[label] * 2 for label in original_labels]))
        image_files = []
        for _id in ids:
            split_id = _id.split('_')
            filename_base = f'./mnt/inputs/{mode}/{split_id[0]}/' \
                            f'Plate{split_id[1]}/{split_id[2]}'
            for site in [1, 2]:
                _image_files = []
                for w in [1, 2, 3, 4, 5, 6]:
                    _image_files.append(f'{filename_base}_s{site}_w{w}.png')
                image_files.append(_image_files)
        return image_files, labels

    def _load_one_img(self, image_file_set):
        _images = []
        for image_file in image_file_set:
            # 0 means gray scale
            _images.append(cv2.imread(image_file, 0))
        loaded_image = np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6)
        return loaded_image

        #     def _augmentation(self, img):
        #         # -------
        #         def _albumentations(mode, visualize):
        #             aug_list = []
        #
        #             aug_list.append(
        #                 Resize(
        #                     IMAGE_SIZE,
        #                     IMAGE_SIZE,
        #                     interpolation=cv2.INTER_CUBIC,
        #                     p=1.0)
        #             )
        #
        #             if mode == "train":  # use data augmentation only with train mode
        #                 aug_list.append(HorizontalFlip(p=0.5))
        #
        #                 # if you want to use additional augmentation, add operations like below.
        #                 # albumentations: [https://github.com/albu/albumentations]
        #                 """
        #                 aug_list.append(
        #                     ShiftScaleRotate(
        #                         p=1.0, shift_limit=0.0625, scale_limit=0.2, rotate_limit=15
        #                     )
        #                 )
        #                 aug_list.append(RandomBrightnessContrast(p=0.5))
        #                 aug_list.append(
        #                     HueSaturationValue(
        #                         p=0.5, hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20
        #                     )
        #                 )
        #                 """
        #
        #             if not visualize:
        #                 aug_list.append(
        #                     Normalize(
        #                         p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #                     )  # rgb
        #                 )  # based on imagenet
        #
        #             return Compose(aug_list, p=1.0)
        #
        #         def _cutout(img):
        #             # [https://arxiv.org/pdf/1708.04552.pdf]
        #             mask_value = [
        #                 int(np.mean(img[:, :, 0])),
        #                 int(np.mean(img[:, :, 1])),
        #                 int(np.mean(img[:, :, 2])),
        #             ]
        #
        #             mask_size_v = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)
        #             mask_size_h = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)
        #
        #             cutout_top = np.random.randint(
        #                 0 - mask_size_v // 2, IMAGE_SIZE - mask_size_v
        #             )
        #             cutout_left = np.random.randint(
        #                 0 - mask_size_h // 2, IMAGE_SIZE - mask_size_h
        #             )
        #             cutout_bottom = cutout_top + mask_size_v
        #             cutout_right = cutout_left + mask_size_h
        #
        #             if cutout_top < 0:
        #                 cutout_top = 0
        #
        #             if cutout_left < 0:
        #                 cutout_left = 0
        #
        #             img[cutout_top:cutout_bottom, cutout_left:cutout_right, :] = mask_value
        #
        #             return img
        #         # -------
        #
        #         img = _albumentations(self.mode, self.visualize)(image=img)["image"]
        #         if (
        #             self.mode == "train"
        #             and np.random.uniform() >= 0.5  # 50%
        #         ):
        #             img = _cutout(img)
        #
        #         return img
