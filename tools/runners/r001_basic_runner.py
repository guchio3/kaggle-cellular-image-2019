import datetime
import time
from glob import glob
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets import CellularImageDataset
from ..models import resnet18
from ..schedulers import pass_scheduler
from ..utils.logs import sel_log, send_line_notification


class Runner(object):
    def __init__(self, config, args, logger):
        # set args info
        self.exp_id = args['exp_id']
        self.checkpoint = args['checkpoint']

        # set config info, and build
        self.exp_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.max_epoch = config['max_epoch']
        self.fobj = self._get_fobj(config['fobj'])
        self.optimizer, self.model = self._build_model(
            config['model'], config['optimizer'])
        scheduler = self._get_scheduler(
            config['scheduler']['scheduler_type'], self.max_epoch)
        self.sampler_type = config['sampler']['sampler_type']
        self.histries = {
            'train_loss': [],
            'valid_loss': [],
            'valid_acc': [],
        }

    def _get_fobj(self, fobj_type):
        if fobj_type == 'ce':
            fobj = nn.CrossEntropyLoss()
        else:
            raise Exception(f'invalid fobj_type: {fobj_type}')
        return fobj

    def _get_model(self, model_type, pretrained):
        if model_type == 'resnet18':
            model = resnet18.Network(pretrained, 1108)
        else:
            raise Exception(f'invalid model_type: {model_type}')
        return model.to(self.device)

    def _get_optimizer(self, optim_type, lr, model):
        if optim_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True,
            )
        elif optim_type == 'adam':
            optimizer = optim.Adam(
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
        elif scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    int(max_epoch * 0.8),
                    int(max_epoch * 0.9)
                ],
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            # scheduler examples:
            #     [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
            # if you want to use cosine annealing, use below scheduler.
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epoch, eta_min=0.0001
            )
        else:
            raise Exception(f'invalid scheduler_type: {scheduler_type}')
        return scheduler

    def _get_sampler(self, dataset, mode, sampler_type):
        if mode == "train":
            if sampler_type == 'random':
                sampler = torch.utils.data.sampler.RandomSampler(
                    data_source=dataset.image_files)
            elif sampler_type == 'weighted':
                sampler = torch.utils.data.sampler.WeightedRandomSampler(
                    dataset.class_weight, dataset.__len__()
                )
            else:
                raise Exception(f'invalid sampler_type: {sampler_type}')
        else:  # valid, test
            sampler = torch.utils.data.sampler.SequentialSampler(
                data_source=dataset.image_files)
        return sampler

    def _build_loader(self, mode, ids, root_dir, augment):
        dataset = CellularImageDataset(mode, ids, root_dir, augment)
        sampler = self._get_sampler(dataset, mode, self.sampler_type)
        drop_last = True if mode == 'train' else False
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=cpu_count(),
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    def _calc_accuracy(self, preds, labels):
        return (preds == labels).mean()
#     def _calc_accuracy(self, preds, labels):
#         score = 0
#         for (pred, label) in zip(preds, labels):
#             if pred == label:
#                 score += 1
#         return score / len(labels)

    def _train_loop(self, loader):
        self.model.train()
        running_loss = 0

        for (images, labels) in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model.forward(images)

            train_loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            train_loss.backward()

            self.optimizer.step()

            running_loss += train_loss.item()

        train_loss = running_loss / len(loader)

        return train_loss

    def _valid_loop(self, loader):
        self.model.eval()
        running_loss = 0

        valid_preds, valid_labels = [], []
        for (images, labels) in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model.forward(images)
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

    def _test_loop(self, loader):
        self.model.eval()

        test_preds = []

        for (images, labels) in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model.forward(images)
            _, predicted = torch.max(outputs.data, 1)

            test_preds.append(predicted.cpu())

        test_preds = torch.cat(test_preds)

        return test_preds

    def _load_checkpoint(self, cp_filename):
        return torch.load(cp_filename)

    def _save_checkpoint(self, current_epoch, val_loss, val_acc):
        # pth means pytorch
        cp_filename = f'./mnt/checkpoints/{self.exp_id}/' \
            f'epoch_{current_epoch}_{val_loss:.5f}' \
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

    def _load_best_model(self):
        best_epoch = np.argmax(self.histories['val_acc'])
        best_cp_filename = glob(
            f'./mnt/checkpoints/{self.exp_id}/epoch_{best_epoch}*')[0]
        best_checkpoint = self._load_checkpoint(best_cp_filename)
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        best_loss = self.histories['val_loss'][best_epoch]
        best_acc = self.histories['val_acc'][best_epoch]
        return best_loss, best_acc

    # -------

    def train_model(self):
        # TODO
        train_loader = self._build_loader(mode="train")
        valid_loader = self._build_loader(mode="valid")

        # load and apply checkpoint if needed
        if self.checkpoint:
            sel_log(f'loading checkpoint from {self.checkpoint} ...',
                    self.logger)
            checkpoint = self.load_checkpoint(self.checkpoint)
            current_epoch = checkpoint['current_epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.histories = checkpoint['histories']
            iter_epochs = range(current_epoch + 1, self.max_epoch + 1, 1)
        else:
            iter_epochs = range(1, self.max_epoch + 1, 1)

        epoch_start_time = time.time()
        sel_log('start tringing !', self.logger)
        for current_epoch in iter_epochs:
            start_time = time.time()
            train_loss = self._train_loop(train_loader)
            valid_loss, valid_acc = self._valid_loop(valid_loader)

            sel_log(
                f'epoch: {current_epoch} / '
                + f'train loss: {train_loss:.5f} / '
                + f'valid loss: {valid_loss:.5f} / '
                + f'valid acc: {valid_acc:.5f} / '
                + f'lr: {self.optimizer.param_groups[0]["lr"]:.5f} / '
                + f'time: {int(time.time()-start_time)}sec', self.logger)

            self.histries['train_loss'].append(train_loss)
            self.histries['valid_loss'].append(valid_loss)
            self.histries['valid_acc'].append(valid_acc)

            self.scheduler.step()
            self._save_checkpoint(current_epoch, valid_loss, valid_acc)

        self.trn_time = int(time.time() - epoch_start_time) // 60

    def make_submission_file(self):
        test_loader = self._build_loader(mode="test")
        best_loss, best_acc = self._load_best_model()
        test_preds = self._test_loop(test_loader)

        submission_df = pd.read_csv(
            './mnt/inputs/origin/sample_submission.csv')
        submission_df['label'] = test_preds
        filename_base = f'{self.exp_id}_{self.exp_time}_' \
            f'{best_loss:.5f}_{best_acc:.5f}'
        sub_filename = f'./mnt/submissions/{filename_base}_sub.csv'
        submission_df.to_csv(sub_filename, index=False)

        sel_log(f'Saved submission file to {sub_filename} !', self.logger)
        line_message = f'Finished the whole pipeline ! \n' \
            f'Training time : {self.trn_time} min \n' \
            f'Best valid loss : {best_loss:.5f} \n' \
            f'Best valid acc : {best_acc:.5f}'
        send_line_notification(line_message)

    def plot_history(self):
        raise NotImplementedError()
