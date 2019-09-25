import datetime
import gc
import os
import random
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold as gkf
from sklearn.model_selection import StratifiedKFold as skf
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from adabound import AdaBound

from ..datasets import CellularImageDataset, ImagesDS
from ..models import (densenet121_metric, densenet161_metric, densenet169_metric, densenet201_metric, resnext101_metric, efficientnetb2,
                      efficientnetb2_metric, efficientnetb2_metric_bn,
                      efficientnetb2_metric_larger, efficientnetb4,
                      efficientnetb5, efficientnetb5_metric, efficientnetb6_metric, efficientnetb7, resnet18)
from ..schedulers import CosineAnnealingWarmUpRestarts as cawur
from ..schedulers import pass_scheduler
from ..utils.logs import sel_log, send_line_notification
from ..utils.splittings import CellwiseStratifiedKFold as cskf
from ..utils.splittings import ExperimentwiseSplit as ews
from ..utils.splittings import WholeDataSplit as wds


def cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


class Runner(object):
    def __init__(self, config, args, logger):
        # set args info
        # -1 for just prediction
        self.trn_time = -1
        self.exp_id = args.exp_id
        self.checkpoint = args.checkpoint
        self.debug = args.debug

        # set config info, and build
        self.exp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.max_epoch = config['max_epoch']
        self.fobj = self._get_fobj(config['fobj'])
        if 'easy_margin' in config['model']:
            self.easy_margin = config['model']['easy_margin']
        else:
            self.easy_margin = True
        self.model, self.optimizer = self._build_model(
            config['model'], config['optimizer'])
        self.scheduler = self._get_scheduler(
            config['scheduler']['scheduler_type'], self.max_epoch)
        self.sampler_type = config['sampler']['sampler_type']
        self.split_type = config['split']['split_type']
        self.split_num = config['split']['split_num']
        if 'augment' in config:
            self.augment = config['augment']
        else:
            self.augment = []
        if 'ttas' in config:
            self.ttas = config['ttas']
        else:
            self.ttas = ['original', ]
        self.metric = True if 'metric' in config['model']['model_type'] else False
        self.logger = logger
        self.histories = {
            'train_loss': [],
            'valid_loss': [],
            'valid_acc': [],
        }
        self.base_weight = args.base_weight
        if args.base_weight:
            if args.base_weight == 'best_all':
                args.base_weight, _, _ = self._search_best_filename('ALL')
            checkpoint = torch.load(args.base_weight)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def _get_fobj(self, fobj_type):
        if fobj_type == 'ce':
            fobj = nn.CrossEntropyLoss().to(self.device)
        elif fobj_type == 'sce':
            fobj = cross_entropy
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, pretrained):
        if model_type == 'resnet18':
            model = resnet18.Network(pretrained, 1108)
        elif model_type == 'efficientnetb2':
            model = efficientnetb2.Network(pretrained, 1108)
        elif model_type == 'efficientnetb4':
            model = efficientnetb4.Network(pretrained, 1108)
        elif model_type == 'efficientnetb5':
            model = efficientnetb5.Network(pretrained, 1108)
        elif model_type == 'efficientnetb7':
            model = efficientnetb7.Network(pretrained, 1108)
        elif model_type == 'efficientnetb2_metric':
            model = efficientnetb2_metric.Network(pretrained, 1108, self.easy_margin)
        elif model_type == 'efficientnetb2_metric_bn':
            model = efficientnetb2_metric_bn.Network(pretrained, 1108)
        elif model_type == 'efficientnetb2_metric_larger':
            model = efficientnetb2_metric_larger.Network(pretrained, 1108)
        elif model_type == 'efficientnetb5_metric':
            model = efficientnetb5_metric.Network(pretrained, 1108, self.easy_margin)
        elif model_type == 'efficientnetb6_metric':
            model = efficientnetb6_metric.Network(pretrained, 1108, self.easy_margin)
        elif model_type == 'densenet121_metric':
            model = densenet121_metric.Network(pretrained, 1108)
        elif model_type == 'densenet161_metric':
            model = densenet161_metric.Network(pretrained, 1108)
        elif model_type == 'densenet169_metric':
            model = densenet169_metric.Network(pretrained, 1108)
        elif model_type == 'densenet201_metric':
            model = densenet201_metric.Network(pretrained, 1108)
        elif model_type == 'resnext101_metric':
            model = resnext101_metric.Network(pretrained, 1108)
        else:
            raise Exception(f'invalid model_type: {model_type}')
        # return model.to(self.device)
        return torch.nn.DataParallel(model.to(self.device))

    def _get_optimizer(self, optim_type, lr, model):
        if optim_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                # weight_decay=1e-4,
                nesterov=True,
            )
        elif optim_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
            )
        elif optim_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.9,
            )
        elif optim_type == 'adabound':
            optimizer = AdaBound(
                model.parameters(),
                lr=lr,
            )
        else:
            raise Exception(f'invalid optim_type: {optim_type}')
        return optimizer

    def _build_model(self, m_config, o_config):
        model = self._get_model(
            m_config['model_type'],
            m_config['pretrained'],
        )
        optimizer = self._get_optimizer(
            o_config['optim_type'],
            o_config['lr'],
            model,
        )
        return model, optimizer

    def _get_scheduler(self, scheduler_type, max_epoch):
        if scheduler_type == 'pass':
            scheduler = pass_scheduler()
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=2,
                gamma=0.9,
            )
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    int(max_epoch * 0.5),
                    int(max_epoch * 0.9)
                ],
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            # scheduler examples:
            #     [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
            # if you want to use cosine annealing, use below scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epoch, eta_min=0.00001
            )
        elif scheduler_type == 'cawur':
            scheduler = cawur(
                self.optimizer, T_max=max_epoch, eta_min=0.0001
            )
        else:
            raise Exception(f'invalid scheduler_type: {scheduler_type}')
        return scheduler

    def _get_sampler(self, dataset, mode, sampler_type):
        # if mode == "train":
        if True:
            if sampler_type == 'random':
                print('random sampler !!')
                sampler = torch.utils.data.sampler.RandomSampler(
                    data_source=dataset)
                # data_source=dataset.image_files)
            elif sampler_type == 'weighted':
                sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    dataset.class_weight, dataset.__len__()
                )
            elif sampler_type == 'sequential':
                print('sequential sampler !!')
                sampler = torch.utils.data.sampler.SequentialSampler(
                    data_source=dataset)
            else:
                raise Exception(f'invalid sampler_type: {sampler_type}')
        else:  # test
            sampler = torch.utils.data.sampler.SequentialSampler(
                data_source=dataset)
            # data_source=dataset.image_files)
        return sampler

    def _build_loader(self, mode, ids, augment, batch_size=None, valid=False):
        dataset = CellularImageDataset(mode, ids, augment)
        # dataset = ImagesDS(ids, './mnt/inputs/', mode)
        sampler = self._get_sampler(dataset, mode, self.sampler_type)
        drop_last = True if mode == 'train' and not valid else False
#        shuffle = True if mode == 'train' else False
        if not batch_size:
            # specify for evaluation
            batch_size = self.batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=os.cpu_count(),
            # num_workers=0,
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
            #            shuffle=shuffle,
        )
        return loader

    def _calc_accuracy(self, preds, labels):
        return (preds == labels).numpy().mean()
#     def _calc_accuracy(self, preds, labels):
#         score = 0
#         for (pred, label) in zip(preds, labels):
#             if pred == label:
#                 score += 1
#         return score / len(labels)

    def _train_loop(self, loader):
        self.model.train()
        running_loss = 0

        for (ids, images, labels, labels_dist, means, stds) in tqdm(loader):
            images, labels = images.to(
                self.device, dtype=torch.float), labels.to(
                self.device)
#             if (
#                     'normalize' in self.augment
#                     or 'normalize_exp' in self.augment
#                     or 'normalize_plate_exp' in self.augment
#             ):
#                 means = means.to(self.device, dtype=torch.float)
#                 means = means.reshape(
#                     self.batch_size, 6, 1, 1).expand(
#                     self.batch_size, 6, 512, 512)
#                 stds = stds.to(self.device, dtype=torch.float)
#                 stds = stds.reshape(
#                     self.batch_size, 6, 1, 1).expand(
#                     self.batch_size, 6, 512, 512)
#                 images -= means
#                 images /= stds

            if self.metric and 'mixup' in self.augment:
                labels_dist = labels_dist.to(self.device, dtype=torch.float)
                outputs = self.model.forward(images, labels_dist)
            elif self.metric:
                outputs = self.model.forward(images, labels)
            else:
                outputs = self.model.forward(images)

            if 'mixup' in self.augment:
                train_loss = self.fobj(outputs, labels_dist)
            else:
                train_loss = self.fobj(outputs, labels)

            self.optimizer.zero_grad()
            train_loss.backward()

            self.optimizer.step()

            running_loss += train_loss.item()

        train_loss = running_loss / len(loader)
#        images.detach()
#        labels.detach()

        return train_loss

    def _valid_loop(self, loader):
        self.model.eval()
        running_loss = 0

        with torch.no_grad():
            valid_preds, valid_labels = [], []
            for (ids, images, labels, labels_dist, means, stds) in tqdm(loader):
                images, labels = images.to(
                    self.device, dtype=torch.float), labels.to(
                    self.device)
#                 if (
#                         'normalize' in self.augment
#                         or 'normalize_exp' in self.augment
#                         or 'normalize_plate_exp' in self.augment
#                 ):
#                     means = means.to(self.device, dtype=torch.float)
#                     means = means.reshape(
#                         self.batch_size, 6, 1, 1).expand(
#                         self.batch_size, 6, 512, 512)
#                     stds = stds.to(self.device, dtype=torch.float)
#                     stds = stds.reshape(
#                         self.batch_size, 6, 1, 1).expand(
#                         self.batch_size, 6, 512, 512)
#                     images -= means
#                     images /= stds

                outputs = self.model.forward(images)
                if 'mixup' in self.augment:
                    labels_dist = labels_dist.to(self.device, dtype=torch.float)
                    valid_loss = self.fobj(outputs, labels_dist)
                else:
                    valid_loss = self.fobj(outputs, labels)
                running_loss += valid_loss.item()

                _, predicted = torch.max(outputs.data, 1)

                valid_preds.append(predicted.cpu())
                valid_labels.append(labels.cpu())

            valid_loss = running_loss / len(loader)

            valid_preds = torch.cat(valid_preds)
            valid_labels = torch.cat(valid_labels)
            valid_accuracy = self._calc_accuracy(
                valid_preds, valid_labels
            )

        return valid_loss, valid_accuracy

    def _test_loop(self, loader, ttas):
        self.model.eval()

        test_ids = []
        test_preds = []

        sel_log('predicting ...', self.logger)
        with torch.no_grad():
            for tta in ttas:
                sel_log(f'tta: {tta}', self.logger)
                loader.dataset.tta = tta
                for (ids, images, labels, labels_dist,
                     means, stds) in tqdm(loader):
                    images, labels = images.to(
                        self.device, dtype=torch.float), labels.to(
                        self.device)
#                     if (
#                             'normalize' in self.augment
#                             or 'normalize_exp' in self.augment
#                             or 'normalize_plate_exp' in self.augment
#                     ):
#                         _batch_size = images.shape[0]
#                         means = means.to(self.device, dtype=torch.float)
#                         means = means.reshape(
#                             _batch_size, 6, 1, 1).expand(
#                             _batch_size, 6, 512, 512)
#                         stds = stds.to(self.device, dtype=torch.float)
#                         stds = stds.reshape(
#                             _batch_size, 6, 1, 1).expand(
#                             _batch_size, 6, 512, 512)
#                         images -= means
#                         images /= stds

                    outputs = self.model.forward(images)
                    sm_outputs = softmax(outputs, dim=1)
    #                sm_outputs = torch.mean(torch.stack(
    #                    [sm_outputs[i::AUGNUM] for i in range(AUGNUM)], dim=2), dim=2)
    #                _, predicted = torch.max(sm_outputs.data, 1)

                    test_ids.append(ids)
                    test_preds.append(sm_outputs.cpu())

            test_ids = np.concatenate(test_ids)
            test_preds = torch.cat(test_preds).numpy()

        res_df = pd.DataFrame([test_ids, test_preds]).T
        res_df.columns = ['test_id', 'y_pred']
        res_ids = []
        res_raws = []
        res_preds = []
        for i, grp_df in res_df.groupby('test_id'):
            res_ids.append(i)
            res_raw = np.mean(grp_df['y_pred'].values, axis=0)
            res_raws.append(res_raw)
            y_pred = np.argmax(res_raw)
            res_preds.append(y_pred)

        return res_ids, res_preds, res_raws

    def _load_checkpoint(self, cp_filename):
        checkpoint = torch.load(cp_filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.histories = checkpoint['histories']
        return checkpoint

    def _save_checkpoint(self, current_epoch, val_loss, val_acc, cell_type):
        if not os.path.exists(f'./mnt/checkpoints/{self.exp_id}'):
            os.makedirs(f'./mnt/checkpoints/{self.exp_id}')
        # pth means pytorch
        cp_filename = f'./mnt/checkpoints/{self.exp_id}/' \
            f'cell_type_{cell_type}_epoch_{current_epoch}_{val_loss:.5f}' \
            f'_{val_acc:.5f}_checkpoint.pth'
        cp_dict = {
            'current_epoch': current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'histories': self.histories,
        }
        sel_log(f'now saving checkpoint to {cp_filename} ...', self.logger)
        torch.save(cp_dict, cp_filename)

    def _search_best_filename(self, cell_type):
        best_loss = np.inf
        best_acc = -1
        best_filename = ''
        for filename in glob(f'./mnt/checkpoints/{self.exp_id}/*'):
            if cell_type not in filename:
                continue
            split_filename = filename.split('/')[-1].split('_')
            temp_loss = float(split_filename[5])
            temp_acc = float(split_filename[6])
            # if temp_loss < best_loss:
            if temp_acc >= best_acc:
                best_loss = temp_loss
                best_acc = temp_acc
                best_filename = filename
        return best_filename, best_loss, best_acc

    def _load_best_model(self, cell_type):
        best_cp_filename, best_loss, best_acc = self._search_best_filename(
            cell_type)
        sel_log(f'the best file is {best_cp_filename} !', self.logger)
        _ = self._load_checkpoint(best_cp_filename)
        return best_loss, best_acc

    def _trn_val_split(self, split_type, split_num, cell_type):
        trn_df = pd.read_csv('./mnt/inputs/origin/train.csv.zip')

        if split_type == 'gkf':
            fold = gkf(split_num).split(
                trn_df['id_code'], trn_df['sirna'], trn_df['well'])
        elif split_type == 'skf':
            fold = skf(split_num, shuffle=True, random_state=71)\
                .split(trn_df['id_code'], trn_df['sirna'])
        elif split_type == 'cskf':
            fold = cskf(
                trn_df,
                trn_df['sirna'],
                split_num,
                shuffle=True,
                random_state=71)
        elif split_type == 'ews':
            fold = ews(
                trn_df,
                trn_df['sirna'],
                split_num,
                shuffle=True,
                random_state=71)
        elif split_type == 'wds':
            fold = wds(
                trn_df,
                trn_df['sirna'],
                split_num,
                shuffle=True,
                random_state=71,
                cell_type=cell_type)
        else:
            raise Exception(f'invalid split type: {split_type}')
        if cell_type not in ['ALL', 'HEPG2', 'U2OS', 'HUVEC', 'RPE']:
            raise Exception(f'invalid cell type {cell_type}')
        for trn_idx, val_idx in fold:
            if cell_type != 'ALL':
                _trn_df = trn_df.iloc[trn_idx]
                trn_ids = _trn_df[_trn_df.experiment.str.contains(
                    cell_type)].id_code
                _val_df = trn_df.iloc[val_idx]
                val_ids = _val_df[_val_df.experiment.str.contains(
                    cell_type)].id_code
            else:
                trn_ids = trn_df.iloc[trn_idx].id_code
                val_ids = trn_df.iloc[val_idx].id_code
            break
        return trn_ids, val_ids

    def _get_test_ids(self, cell_type='ALL'):
        tst_df = pd.read_csv('./mnt/inputs/origin/test.csv')
        if cell_type not in ['ALL', 'HEPG2', 'U2OS', 'HUVEC', 'RPE']:
            raise Exception(f'invalid cell type {cell_type}')
        if cell_type != 'ALL':
            tst_df = tst_df[tst_df.experiment.str.contains(cell_type)]
        tst_ids = tst_df.id_code
        return tst_ids

    def _warmup_setting(self, epoch):
        if self.base_weight:
            return
        if epoch < 3:
            # for name, child in self.model.named_children():
            for name, child in self.model.module.named_children():
                if 'fc' in name or 'arc' in name or '_conv_stem' in name or 'bn00' in name:
                    sel_log(name + ' is unfrozen', self.logger)
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    sel_log(name + ' is frozen', self.logger)
                    for param in child.parameters():
                        param.requires_grad = False
        elif epoch == 3:
            sel_log("Turn on all the layers", self.logger)
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    # -------

    def train_model(self, cell_type='ALL'):
        trn_ids, val_ids = self._trn_val_split(
            self.split_type, self.split_num, cell_type)
        if self.debug:
            trn_ids = trn_ids[:300]
            val_ids = val_ids[:300]

        train_loader = self._build_loader(
            mode="train", ids=trn_ids, augment=self.augment)
        if 'normalize' in self.augment:
            augment = ['normalize']
        elif 'normalize_exp' in self.augment:
            augment = ['normalize_exp']
        elif 'normalize_plate_exp' in self.augment:
            augment = ['normalize_plate_exp']
        else:
            augment = []
        valid_loader = self._build_loader(
            mode="train", ids=val_ids, augment=augment, valid=True)

        # load and apply checkpoint if needed
        if self.checkpoint:
            sel_log(f'loading checkpoint from {self.checkpoint} ...',
                    self.logger)
            checkpoint = self._load_checkpoint(self.checkpoint)
            current_epoch = checkpoint['current_epoch']
            iter_epochs = range(current_epoch + 1, self.max_epoch + 1, 1)
        else:
            iter_epochs = range(1, self.max_epoch + 1, 1)

        epoch_start_time = time.time()
        sel_log('start trainging !', self.logger)
        for current_epoch in iter_epochs:
            self._warmup_setting(current_epoch)
            start_time = time.time()
            train_loss = self._train_loop(train_loader)
            valid_loss, valid_acc = self._valid_loop(valid_loader)

            sel_log(
                f'cell_type: {cell_type} / '
                + f'epoch: {current_epoch} / '
                + f'train loss: {train_loss:.5f} / '
                + f'valid loss: {valid_loss:.5f} / '
                + f'valid acc: {valid_acc:.5f} / '
                + f'lr: {self.optimizer.param_groups[0]["lr"]:.5f} / '
                + f'time: {int(time.time()-start_time)}sec', self.logger)

            self.histories['train_loss'].append(train_loss)
            self.histories['valid_loss'].append(valid_loss)
            self.histories['valid_acc'].append(valid_acc)

            self.scheduler.step()
            self._save_checkpoint(
                current_epoch, valid_loss, valid_acc, cell_type)

        self.trn_time = int(time.time() - epoch_start_time) // 60
        del train_loader, valid_loader
        gc.collect()

    def make_submission_file(self, cell_type='ALL', sub_filename=None):
        tst_ids = self._get_test_ids(cell_type)
        if self.debug:
            tst_ids = tst_ids[:300]
        if 'normalize' in self.augment:
            augment = ['normalize']
        elif 'normalize_exp' in self.augment:
            augment = ['normalize_exp']
        elif 'normalize_plate_exp' in self.augment:
            augment = ['normalize_plate_exp']
        else:
            augment = []
        test_loader = self._build_loader(
            mode="test", ids=tst_ids, augment=augment, batch_size=self.batch_size)
        best_loss, best_acc = self._load_best_model(cell_type)
        test_ids, test_preds, test_raw_preds = self._test_loop(
            test_loader, self.ttas)

        if sub_filename:
            submission_df = pd.read_csv(sub_filename)
            sub_raw_filename = sub_filename.replace(
                '_sub', '_sub_raw').replace(
                '.csv', '.pkl')
            submission_raw_df = pd.read_pickle(sub_raw_filename)
        else:
            submission_df = pd.read_csv(
                './mnt/inputs/origin/sample_submission.csv')
            submission_raw_df = pd.read_csv(
                './mnt/inputs/origin/sample_submission.csv')
            submission_raw_df = submission_raw_df.rename(
                columns={'sirna': 'raw_pred'})
            submission_raw_df['raw_pred'] = None
        submission_df = submission_df.set_index('id_code')
        submission_df.loc[test_ids, 'sirna'] = test_preds
        submission_df = submission_df.reset_index()
        submission_raw_df = submission_raw_df.set_index('id_code')
        submission_raw_df.loc[test_ids, 'raw_pred'] = test_raw_preds
        submission_raw_df = submission_raw_df.reset_index()
        if not sub_filename:
            filename_base = f'{self.exp_id}_{self.exp_time}_' \
                f'{best_loss:.5f}_{best_acc:.5f}'
            sub_filename = f'./mnt/submissions/{filename_base}_sub.csv'
            sub_raw_filename = f'./mnt/submissions/{filename_base}_sub_raw.pkl'
        submission_df.to_csv(sub_filename, index=False)
        submission_raw_df.to_pickle(sub_raw_filename)

        sel_log(f'Saved submission file to {sub_filename} !', self.logger)
        return sub_filename

    def plot_history(self):
        raise NotImplementedError()
