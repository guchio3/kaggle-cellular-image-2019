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
                            RandomRotate90, RandomSizedCrop, Resize, Rotate,
                            ShiftScaleRotate, VerticalFlip)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .utils.logs import sel_log

# sys.path.append('../tools/utils/rxrx1-utils')

IMAGE_SIZE = 512
RESIZE_IMAGE_SIZE = 256


def _load_imgs_from_ids(id_pair, mode):
    _id, label, plate = id_pair
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
            if os.path.isfile(f'{filename_base}_s{site}_w{w}.png'):
                img = cv2.imread(f'{filename_base}_s{site}_w{w}.png', 0)
            else:
                filename_base = f'./mnt/inputs/test/{split_id[0]}/' \
                                f'Plate{split_id[1]}/{split_id[2]}'
                img = cv2.imread(f'{filename_base}_s{site}_w{w}.png', 0)
            _images.append(img)
#        images.append(
#            np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6))
        res_id_pairs.append(
            [_id, np.array(_images).transpose(1, 2, 0), label, site, plate])
        #    [_id, np.array(_images).reshape(IMAGE_SIZE, IMAGE_SIZE, 6), label, site])
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
        self.tta = None
        self.len = None

        if mode == "test":
            tst_df = pd.read_csv(
                './mnt/inputs/origin/test.csv').set_index('id_code')
            labels = [0] * len(ids)
            plates = tst_df.loc[ids]['plate'].values
        else:  # train or valid
            trn_df = pd.read_csv(
                './mnt/inputs/origin/merged_df.csv').set_index('id_code')
#                './mnt/inputs/origin/train.csv.zip').set_index('id_code')
            labels = trn_df.loc[ids]['sirna'].values
            plates = trn_df.loc[ids]['plate'].values
        self.ids, self.images, self.labels, self.sites, self.plates = self._parse_ids(
            mode, ids, labels, plates)
        self.stats_df = pd.read_csv('./mnt/inputs/origin/pixel_stats.csv.zip')
        self.agg_stats_df = self.stats_df.groupby(['experiment', 'channel']).aggregate({
            'mean': ['mean'], 'std': ['mean']})
        self.plate_agg_stats_df = self.stats_df.groupby(['experiment', 'plate', 'channel']).aggregate({
            'mean': ['mean'], 'std': ['mean']})

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
        id_code = self.ids[idx]
        site = self.sites[idx]
        plate = self.plates[idx]

        experiment = id_code.split('_')[0]
        if 'normalize' in self.augment:
            means = torch.tensor(self.stats_df.query(
                f'id_code == "{id_code}" and site == {site}')['mean'].values)
            stds = torch.tensor(self.stats_df.query(
                f'id_code == "{id_code}" and site == {site}')['std'].values)
        elif 'normalize_exp' in self.augment:
            means = torch.tensor(
                self.agg_stats_df['mean']['mean'].loc[experiment].values)
            means = (means / 255.).tolist()
    #        means = (means / 255.).tolist()
            stds = torch.tensor(
                self.agg_stats_df['std']['mean'].loc[experiment].values)
            stds = (stds / 255.).tolist()
    #        stds = (stds / 255.).tolist()
        elif 'normalize_plate_exp' in self.augment:
            means = torch.tensor(
                self.plate_agg_stats_df['mean']['mean'].loc[experiment].loc[plate].values)
            means = (means / 255.).tolist()
            stds = torch.tensor(
                self.plate_agg_stats_df['std']['mean'].loc[experiment].loc[plate].values)
            stds = (stds / 255.).tolist()
        else:
            means = None
            stds = None

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img, label_dist = self._augmentation(
            img, self.labels[idx], id_code, site, means, stds)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        if 'resize' in self.augment:
            assert img.shape == (6, RESIZE_IMAGE_SIZE, RESIZE_IMAGE_SIZE)
        else:
            assert img.shape == (6, IMAGE_SIZE, IMAGE_SIZE)

        return (self.ids[idx], torch.tensor(img, dtype=torch.float),
                torch.tensor(self.labels[idx]), torch.tensor(label_dist), means, stds)
        # torch.tensor(self.labels[idx]), label_dist, means, stds)

    def reset_ids(self, ids):
        self.len = len(ids)
        self.id_converter = pd.Series(self.ids, name='id_name')\
            .reset_index()\
            .set_index('id_name')\
            .loc[ids]\
            .reset_index(drop=True)\
            .to_dict()

    def _parse_ids(self, mode, ids, original_labels, plates):
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
            iter_func = partial(
                _load_imgs_from_ids,
                mode=mode)
            imap = p.imap_unordered(iter_func, list(
                zip(ids, original_labels, plates)))
            res_id_pairs = list(tqdm(imap, total=len(ids)))
            p.close()
            p.join()
            gc.collect()
        res_id_pairs = list(chain.from_iterable(res_id_pairs))

        ids, images, labels, sites, plates = [], [], [], [], []
        for res_id_pair in res_id_pairs:
            ids.append(res_id_pair[0])
            images.append(res_id_pair[1])
            labels.append(res_id_pair[2])
            sites.append(res_id_pair[3])
            plates.append(res_id_pair[4])

        return ids, images, labels, sites, plates

    def _augmentation(self, img, label, id_code, site, means, stds):
        # -------
        def _albumentations(mode, visualize, means, stds):
            aug_list = []

            if 'resize' in self.augment:
                aug_list.append(Resize(RESIZE_IMAGE_SIZE,
                                       RESIZE_IMAGE_SIZE,
                                       interpolation=cv2.INTER_CUBIC,
                                       p=1.0))

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
                        min_max_height=(256, 512),
                        height=512,
                        width=512,
                        p=0.5,
                    ))
            if (
                'normalize'in self.augment
                or 'normalize_exp'in self.augment
                or 'normalize_plate_exp'in self.augment
            ):
                aug_list.append(Normalize(p=1.0, mean=means, std=stds))

#             #  if not visualize:
#             if 'normalize' in self.augment:
#                 experiment, plate, well = id_code.split('_')
#                 norm_df = self.stats_df.query(
#                         f'experiment == "{experiment}" and '
#                         f'plate == {int(plate)} and '
#                         f'well == "{well}" and '
#                         f'site == {int(site)}'
#                 ).sort_values('channel')
#                 aug_list.append(
#                     Normalize(
#                         p=1.0,
#                         mean=norm_df['mean'].tolist(),
#                         std=norm_df['std'].tolist(),
#                         # mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
#                         # std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
#                     )  # rgb -> 6 channels
#                 )  # based on imagenet

            return Compose(aug_list, p=1.0)

        def _cutout(img):
            # [https://arxiv.org/pdf/1708.04552.pdf]
            mask_value = [
                int(np.mean(img[:, :, 0])),
                int(np.mean(img[:, :, 1])),
                int(np.mean(img[:, :, 2])),
                int(np.mean(img[:, :, 3])),
                int(np.mean(img[:, :, 4])),
                int(np.mean(img[:, :, 5])),
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

        def _mixup(img, label, alpha=0.2):
            l = np.random.beta(alpha, alpha, 1)[0]
            rand_idx = np.random.randint(len(self.labels))
            id_code = self.ids[rand_idx]
            plate = self.plates[rand_idx]
            experiment = id_code.split('_')[0]
            if 'normalize' in self.augment:
                means = torch.tensor(self.stats_df.query(
                    f'id_code == "{id_code}" and site == {site}')['mean'].values)
                stds = torch.tensor(self.stats_df.query(
                    f'id_code == "{id_code}" and site == {site}')['std'].values)
            elif 'normalize_exp' in self.augment:
                means = torch.tensor(
                    self.agg_stats_df['mean']['mean'].loc[experiment].values)
                means = (means / 255.).tolist()
                stds = torch.tensor(
                    self.agg_stats_df['std']['mean'].loc[experiment].values)
                stds = (stds / 255.).tolist()
            elif 'normalize_plate_exp' in self.augment:
                means = torch.tensor(
                    self.plate_agg_stats_df['mean']['mean'].loc[experiment].loc[plate].values)
                means = (means / 255.).tolist()
                stds = torch.tensor(
                    self.plate_agg_stats_df['std']['mean'].loc[experiment].loc[plate].values)
                stds = (stds / 255.).tolist()
            else:
                raise Exception('use normalize_plate_exp')
            rand_img = self.images[rand_idx]
            rand_img = _albumentations(
                'train', self.visualize, means, stds)(
                image=rand_img)["image"]
            rand_label = self.labels[rand_idx]
            img = img * l + rand_img * (1 - l)
            label_dist = np.eye(1108)[label]
            rand_label_dist = np.eye(1108)[rand_label]
            label_dist = label_dist * l + rand_label_dist * (1 - l)
            return img, label_dist
        # -------

        img = _albumentations(self.mode, self.visualize, means, stds)(image=img)["image"]
        if (
            self.mode == "train"
            and 'cutout' in self.augment
            and np.random.uniform() >= 0.5  # 50%
        ):
            img = _cutout(img)

        if (
            self.mode == 'train'
            and 'mixup' in self.augment
            and np.random.uniform() >= 0.
        ):
            img, label_dist = _mixup(img, label)
        elif self.mode == 'train':
            label_dist = np.eye(1108)[label]
        else:
            label_dist = np.zeros(1108)

        if self.tta:
            if self.tta == 'original':
                pass
            elif self.tta == 'rotate90':
                img = RandomRotate90(p=1.0).apply(img, factor=1)
            elif self.tta == 'rotate180':
                img = RandomRotate90(p=1.0).apply(img, factor=2)
            elif self.tta == 'rotate270':
                img = RandomRotate90(p=1.0).apply(img, factor=3)
            elif self.tta == 'flip':
                img = HorizontalFlip(p=1.0).apply(img)
            elif self.tta == 'fliprotate90':
                img = HorizontalFlip(p=1.0).apply(img)
                img = RandomRotate90(p=1.0).apply(img, factor=1)
            elif self.tta == 'fliprotate180':
                img = HorizontalFlip(p=1.0).apply(img)
                img = RandomRotate90(p=1.0).apply(img, factor=2)
            elif self.tta == 'fliprotate270':
                img = HorizontalFlip(p=1.0).apply(img)
                img = RandomRotate90(p=1.0).apply(img, factor=3)
            else:
                raise Exception(f'invalid tta, {self.tta}')

        return img, label_dist


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
