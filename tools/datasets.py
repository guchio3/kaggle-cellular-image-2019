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
from albumentations import (Compose, HorizontalFlip, HueSaturationValue,
                            Normalize, RandomBrightnessContrast,
                            RandomRotate90, Resize, Rotate, ShiftScaleRotate,
                            VerticalFlip, RandomSizedCrop)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .utils.logs import sel_log

# sys.path.append('../tools/utils/rxrx1-utils')

IMAGE_SIZE = 512


def _load_imgs_from_ids(id_pair, mode):
    _id, label = id_pair
    images = []
    split_id = _id.split('_')
    if mode == 'valid':
        mode = 'train'
    filename_base = f'./mnt/inputs/{mode}/{split_id[0]}/' \
                    f'Plate{split_id[1]}/{split_id[2]}'
    res_id_pairs = []
    for site in [1, 2]:
        _images = []
        for w in [1, 2, 3, 4, 5, 6]:
            # 0 means gray scale
            _images.append(
                cv2.imread(f'{filename_base}_s{site}_w{w}.png', 0))
#        images.append(
#            np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6))
        res_id_pairs.append(
            [_id, np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6), label])
    return res_id_pairs


class CellularImageDataset(Dataset):
    def __init__(self, mode, ids, augment,
                 visualize=False, logger=None):
        '''
        ids : id_code
        '''
        self.mode = mode
        self.visualize = visualize
        self.logger = logger
        self.augment = augment
        self.len = None

        if mode == "test":
            labels = [0] * len(ids)
        else:  # train or valid
            labels = pd.read_csv('./mnt/inputs/origin/train.csv.zip')\
                .set_index('id_code').loc[ids]['sirna'].values
        self.ids, self.images, self.labels = self._parse_ids(mode, ids, labels)

        # load validation
        assert len(self.images) == len(self.labels)

    def __len__(self):
        if self.len:
            return self.len
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.len:
            idx = self.id_converter[idx]
        img = self.images[idx]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        assert img.shape == (6, IMAGE_SIZE, IMAGE_SIZE)

        return (self.ids[idx], torch.tensor(img),
                torch.tensor(self.labels[idx]))

    def reset_ids(self, ids):
        self.len = len(ids)
        self.id_converter = pd.Series(self.ids, name='id_name')\
            .reset_index()\
            .set_index('id_name')\
            .loc[ids]\
            .reset_index(drop=True)\
            .to_dict()

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
            res_id_pairs = list(tqdm(imap, total=len(ids)))
            p.close()
            p.join()
            gc.collect()
        res_id_pairs = list(chain.from_iterable(res_id_pairs))

        ids, images, labels = [], [], []
        for res_id_pair in res_id_pairs:
            ids.append(res_id_pair[0])
            images.append(res_id_pair[1])
            labels.append(res_id_pair[2])

        return ids, images, labels

    def _augmentation(self, img):
        # -------
        def _albumentations(mode, visualize):
            aug_list = []

#             aug_list.append(
#                 Resize(
#                     IMAGE_SIZE,
#                     IMAGE_SIZE,
#                     interpolation=cv2.INTER_CUBIC,
#                     p=1.0)
#             )

            if mode == "train":  # use data augmentation only with train mode
                if 'verticalflip' in self.augment:
                    aug_list.append(VerticalFlip(p=0.5))
                if 'horizontalflip' in self.augment:
                    aug_list.append(HorizontalFlip(p=0.5))
                if 'randomrotate90' in self.augment:
                    aug_list.append(RandomRotate90(p=1.))
                if 'rotate' in self.augment:
                    aug_list.append(Rotate(p=0.5))
                if 'brightness'in self.augment:
                    aug_list.append(RandomBrightnessContrast(p=0.5))
                if 'randomsizedcrop'in self.augment:
                    aug_list.append(RandomSizedCrop(
                        min_max_height=(128, 128),
                        height=512,
                        width=512,
                        p=0.5,
                    ))

                # if you want to use additional augmentation, add operations like below.
                # albumentations: [https://github.com/albu/albumentations]
                """
                aug_list.append(
                    ShiftScaleRotate(
                        p=1.0, shift_limit=0.0625, scale_limit=0.2, rotate_limit=15
                    )
                )
                aug_list.append(RandomBrightnessContrast(p=0.5))
                aug_list.append(
                    HueSaturationValue(
                        p=0.5, hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=20
                    )
                )
                """

            #  if not visualize:
            if 'normalize' in self.augment:
                aug_list.append(
                    Normalize(
                        p=1.0,
                        mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
                    )  # rgb -> 6 channels
                )  # based on imagenet

            return Compose(aug_list, p=1.0)

        def _cutout(img):
            # [https://arxiv.org/pdf/1708.04552.pdf]
            mask_value = [
                int(np.mean(img[:, :, 0])),
                int(np.mean(img[:, :, 1])),
                int(np.mean(img[:, :, 2])),
            ]

            mask_size_v = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)
            mask_size_h = int(IMAGE_SIZE * np.random.randint(10, 60) * 0.01)

            cutout_top = np.random.randint(
                0 - mask_size_v // 2, IMAGE_SIZE - mask_size_v
            )
            cutout_left = np.random.randint(
                0 - mask_size_h // 2, IMAGE_SIZE - mask_size_h
            )
            cutout_bottom = cutout_top + mask_size_v
            cutout_right = cutout_left + mask_size_h

            if cutout_top < 0:
                cutout_top = 0

            if cutout_left < 0:
                cutout_left = 0

            img[cutout_top:cutout_bottom, cutout_left:cutout_right, :] = mask_value

            return img
        # -------

        img = _albumentations(self.mode, self.visualize)(image=img)["image"]
#        if (
#            self.mode == "train"
#            and np.random.uniform() >= 0.5  # 50%
#        ):
#            img = _cutout(img)

        return img


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
                    # _image_files.append(f'{filename_base}_s{site}_w{w}.png')
                    _image_files.append(f'{filename_base}_s{site}_w{w}.npy')
                image_files.append(_image_files)
        return image_files, labels

    def _load_one_img(self, image_file_set):
        _images = []
        for image_file in image_file_set:
            # 0 means gray scale
            # _images.append(cv2.imread(image_file, 0))
            _images.append(np.load(image_file))
        loaded_image = np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6)
        return loaded_image


class ImagesDS(Dataset):
    def __init__(self, ids, img_dir, mode='train',
                 channels=[1, 2, 3, 4, 5, 6]):
        df = pd.read_csv(
            f'./mnt/inputs/origin/{mode}.csv').set_index('id_code')
        df = df.loc[ids].reset_index()
        df2 = df.copy()
        df['site'] = 1
        df2['site'] = 2
        df = pd.concat([df, df2], axis=0)
        self.records = df.to_records(index=False)
        self.channels = channels
#        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]

    @staticmethod
    def _load_img_as_tensor(file_name):
        with Image.open(file_name) as img:
            return T.ToTensor()(img)

    def _get_img_path(self, index, channel):
        experiment = self.records[index].experiment
        plate = self.records[index].plate
        well = self.records[index].well
        site = self.records[index].site
        return '/'.join([self.img_dir, self.mode, experiment, f'Plate{plate}',
                         f'{well}_s{site}_w{channel}.png'])

    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]
        img = torch.cat([self._load_img_as_tensor(img_path)
                         for img_path in paths])
        if self.mode == 'train':
            return img, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len
