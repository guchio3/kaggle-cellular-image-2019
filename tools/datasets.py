from glob import glob

import pandas as pd


class CellularImageDataset():
    def __init__(self, mode, visualize=False):
        self.mode = mode
        self.labels = pd.read_csv("../input/train.csv")["label"].values
        self.visualize = visualize

        if mode == "test":
            self.image_files = sorted(glob("../input/images/test/*.png"))
            self.labels = [0] * len(self.image_files)
        else:  # train or valid
            self.image_files = sorted(glob("../input/images/train/*.png"))

            if mode == "train":  # 80% training
                self.image_files = self.image_files[
                    0: int(len(self.image_files) * 0.8)
                ]

                self.labels = self.labels[0: int(len(self.labels) * 0.8)]

            else:  # 20% validation
                self.image_files = self.image_files[int(
                    len(self.image_files) * 0.8):]
                self.labels = self.labels[int(len(self.labels) * 0.8):]

        assert len(self.image_files) == len(self.labels)

        self.class_weight = [(CLASS_WEIGHT[x]) for x in self.labels]

        print("Loading {} images on memory...".format(mode))
        self.images = np.zeros(
            (len(self.image_files), 32, 32, 3)).astype("uint8")

        for i in range(len(self.image_files)):
            self.images[i] = cv2.imread(self.image_files[i])
            if self.visualize and i > 10:
                break

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        img = self._augmentation(img)
        img = img.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)

        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

        return (torch.tensor(img), torch.tensor(self.labels[idx]))

    def _augmentation(self, img):
        # -------
        def _albumentations(mode, visualize):
            aug_list = []

            aug_list.append(
                Resize(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_CUBIC,
                    p=1.0)
            )

            if mode == "train":  # use data augmentation only with train mode
                aug_list.append(HorizontalFlip(p=0.5))

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

            if not visualize:
                aug_list.append(
                    Normalize(
                        p=1.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )  # rgb
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
        if (
            self.mode == "train"
            and np.random.uniform() >= 0.5  # 50%
        ):
            img = _cutout(img)

        return img
