import datetime
import time
from multiprocessing import cpu_count

from ..datasets import CellularImageDataset
from ..models import resnet18
from ..utils.logs import sel_log


class Runner(object):
    def __init__(self, config, logger):
        self.exp_id = config['exp_id']
        self.device = config['device']
        self.fobj = self._get_fobj(config['train']['fobj'])
        self.optimizer, self.model = self._build_model()
        self.histries = {
            'train_loss': [],
            'train_acc': [],
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

    def _build_loader(self, mode, ids, root_dir, augment):
        dataset = CellularImageDataset(mode=mode)

        if mode == "train":
            drop_last_flag = True
            sampler = torch.utils.data.sampler.RandomSampler(
                data_source=dataset.image_files)

            # if you want to sample data based on metric weight, use below
            # sampler.
            """
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                dataset.class_weight, dataset.__len__()
            )
            """

        else:  # valid, test
            drop_last_flag = False
            sampler = torch.utils.data.sampler.SequentialSampler(
                data_source=dataset.image_files)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=cpu_count(),
            worker_init_fn=lambda x: np.random.seed(),
            drop_last=drop_last_flag,
            pin_memory=True,
        )

        return loader

    def _calc_accuracy(self, preds, labels):
        # TODO
        score = 0
        total = 0

        for (pred, label) in zip(preds, labels):
            if pred == label:
                score += CLASS_WEIGHT[label]
            total += CLASS_WEIGHT[label]

        return score / total

    def _get_scheduler(self, scheduler_type, ):
        if scheduler_type == 'pass':
            scheduler =
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[int(EPOCH * 0.8), int(EPOCH * 0.9)], gamma=0.1
        )

        # scheduler examples: [http://katsura-jp.hatenablog.com/entry/2019/01/30/183501]
        # if you want to use cosine annealing, use below scheduler.
        """
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCH, eta_min=0.0001
        )
        """

    def _train_loop(self, loader):
        self.model.train()
        running_loss = 0

        for (images, labels) in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.model.forward(images)
            valid_loss = self.criterion(outputs, labels)
            running_loss += valid_loss.item()

            _, predicted = torch.max(outputs.data, 1)

            valid_preds.append(predicted.cpu())
            valid_labels.append(labels.cpu())

        valid_loss = running_loss / len(loader)

        valid_preds = torch.cat(valid_preds)
        valid_labels = torch.cat(valid_labels)
        valid_weighted_accuracy = self._calc_weighted_accuracy(
            valid_preds, valid_labels
        )

        return valid_loss, valid_weighted_accuracy

    def _test_loop(self, loader):
        self.model.eval()

        test_preds = []

        for (images, labels) in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.model.forward(images)
            _, predicted = torch.max(outputs.data, 1)

            test_preds.append(predicted.cpu())

        test_preds = torch.cat(test_preds)

        return test_preds

    # -------

    def train_model(self):
        # TODO
        train_loader = self._build_loader(mode="train")
        valid_loader = self._build_loader(mode="valid")

        for current_epoch in range(1, EPOCH + 1, 1):
            start_time = time.time()
            train_loss = self._train_loop(train_loader)
            valid_loss, valid_weighted_accuracy = self._valid_loop(
                valid_loader)

            sel_log(
                f'epoch: {current_epoch} / '
                + f'train loss: {train_loss:.5f} / '
                + f'valid loss: {valid_loss:.5f} / '
                + f'valid w-acc: {valid_weighted_accuracy:.5f} / '
                + f'lr: {self.optimizer.param_groups[0]["lr"]:.5f} / '
                + f'time: {int(time.time()-start_time)}sec', self.logger)

            self.train_loss_history.append(train_loss)
            self.valid_loss_history.append(valid_loss)
            self.valid_weighted_accuracy_history.append(
                valid_weighted_accuracy)

            scheduler.step()

    def make_submission_file(self):
        test_loader = self._build_loader(mode="test")
        test_preds = self._test_loop(test_loader)

        submission_df = pd.read_csv("../input/sample_submission.csv")
        submission_df["label"] = test_preds
        submission_df.to_csv("./submission.csv", index=False)

        print("---submission.csv---")
        print(submission_df.head())

    def plot_history(self):
        raise NotImplementedError()
