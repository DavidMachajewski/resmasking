from typing import List, Union
from torchvision import transforms
import numpy as np
import torch
import datetime
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from rmn import RMN, models


class Sample(dict):
    """Accessing dict keys by dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Scale(object):
    def __init__(self, new_size: List[Union[int, int]]):
        self.new_size = new_size

    def __call__(self, sample: Sample) -> Sample:
        h, w = self.new_size
        sample.image = transforms.Resize(size=(h, w))(sample.image)
        return sample


class ToTensor(object):
    def __call__(self, sample: Sample):
        # sample.image = np.asarray(sample.image)[:, :, np.newaxis]
        # from H x W x C to C x H x W
        sample.image = torch.from_numpy(np.asarray(sample.image))
        # sample.image = torch.unsqueeze(sample.image, 2)
        sample.image = sample.image.permute((2, 0, 1))
        # sample.image = torch.unsqueeze(sample.image, 2)
        sample.image = sample.image.float()
        sample.label = int(sample.label)  # torch.Tensor(sample.label)
        return sample


def get_transforms(new_size: List[Union[int, int]]):
    transformer = transforms.Compose([
        Scale(new_size),
        ToTensor()])
    return transformer


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target).float().sum(0)
        acc = correct * 100 / batch_size
    return [acc]


class JAFFETrainer():
    def __init__(self, model, train_set, val_set, test_set, configs):
        """
        :train_set: dataloader of the train set
        :val_set: dataloader of the validation set
        :test_set: dataloader of the test set
        """
        # print start and configs
        #
        # load configurations like the author defines
        self._configs = configs
        self._configs = configs
        self._lr = self._configs["lr"]
        self._batch_size = self._configs["batch_size"]
        self._momentum = self._configs["momentum"]
        self._weight_decay = self._configs["weight_decay"]
        self._distributed = self._configs["distributed"]
        self._num_workers = self._configs["num_workers"]
        self._device = torch.device(self._configs["device"])
        self._max_epoch_num = self._configs["max_epoch_num"]
        self._max_plateau_count = self._configs["max_plateau_count"]
        # model
        self._model = model(in_channels=configs["in_channels"], num_classes=configs["num_classes"])

        self._model.to(self._device)
        # datasets
        self._train_loader = train_set
        self._test_loader = test_set
        self._val_loader = val_set
        # Loss and optimizer
        self._criterion = nn.CrossEntropyLoss().to(self._device)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(),
                                           lr=self._lr,
                                           weight_decay=self._weight_decay)
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self._optimizer,
                                                                     patience=self._configs["plateau_patience"],
                                                                     min_lr=1e-6,
                                                                     verbose=True)

        # training info
        self._start_time = datetime.datetime.now()
        self._start_time = self._start_time.replace(microsecond=0)

        # log_dir = os.path.join(
        #    self._configs["cwd"],
        #    self._configs["log_dir"],
        #    "{}_{}".format(
        #        self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
        #    ),
        # )

        log_dir = os.path.join(
            self._configs["log_dir"],
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )

        self._writer = SummaryWriter(log_dir)
        self._train_loss = []
        self._train_acc = []
        self._val_loss = []
        self._val_acc = []
        self._best_loss = 1e9
        self._best_acc = 0
        self._test_acc = 0.0
        self._plateau_count = 0
        self._current_epoch_num = 0

        # for checkpoints
        #
        #
        # self._checkpoint_dir = self._configs["checkpoint_dir"]
        # self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        #
        #
        self._checkpoint_dir = os.path.join(self._configs["cwd"], "saved/checkpoints")
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir, exist_ok=True)

        self._checkpoint_path = os.path.join(
            self._checkpoint_dir,
            "{}_{}".format(
                self._configs["model_name"], self._start_time.strftime("%Y%b%d_%H.%M")
            ),
        )

    def _train(self):
        # print("training step")
        self._model.train()
        train_loss, train_acc = 0.0, 0.0

        for i, batch in tqdm(enumerate(self._train_loader), total=len(self._train_loader), leave=False):
            # print(f"size of train loader: {len(self._train_loader)}")
            images = batch["image"].to(self._device)  # .to(self._device)
            # print(f"shape of image tensor: {images.shape}")
            targets = batch["label"].to(self._device)  # tensor? .to ...

            outputs = self._model(images)

            loss = self._criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]

            train_loss += loss.item()
            train_acc += acc.item()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        i += 1
        self._train_loss.append(train_loss / i)
        self._train_acc.append(train_acc / i)

    def _val(self):
        print("validation")
        self._model.eval()
        val_loss, val_acc = 0.0, 0.0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self._val_loader), total=len(self._val_loader), leave=False):
                images = batch["image"].to(self._device)  # .cuda(non_blocking=True)
                targets = batch["label"].to(self._device)  # .cuda(non_blocking=True)

                # compute output, measure accuracy and record loss
                outputs = self._model(images)

                loss = self._criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]

                val_loss += loss.item()
                val_acc += acc.item()

            i += 1
            self._val_loss.append(val_loss / i)
            self._val_acc.append(val_acc / i)

    def _increase_epoch_num(self):
        self._current_epoch_num += 1

    def _is_stop(self):
        return (
                self._plateau_count > self._max_plateau_count
                or self._current_epoch_num > self._max_epoch_num
        )

    def _update_training_state(self):
        if self._val_acc[-1] > self._best_acc:
            self._save_weights()
            self._plateau_count = 0
            self._best_acc = self._val_acc[-1]
            self._best_loss = self._val_loss[-1]
        else:
            self._plateau_count += 1

        self._scheduler.step(100 - self._val_acc[-1])

    def _save_weights(self, test_acc=0.0):
        if self._distributed == 0:
            state_dict = self._model.state_dict()
        else:
            state_dict = self._model.module.state_dict()

        state = {
            **self._configs,
            "net": state_dict,
            "best_loss": self._best_loss,
            "best_acc": self._best_acc,
            "train_losses": self._train_loss,
            "val_loss": self._val_loss,
            "train_acc": self._train_acc,
            "val_acc": self._val_acc,
            "test_acc": self._test_acc,
        }
        torch.save(state, self._checkpoint_path)

    def _logging(self):
        consume_time = str(datetime.datetime.now() - self._start_time)

        message = "\nE{:03d}  {:.3f}/{:.3f}/{:.3f} {:.3f}/{:.3f}/{:.3f} | p{:02d}  Time {}\n".format(
            self._current_epoch_num,
            self._train_loss[-1],
            self._val_loss[-1],
            self._best_loss,
            self._train_acc[-1],
            self._val_acc[-1],
            self._best_acc,
            self._plateau_count,
            consume_time[:-7],
        )

        self._writer.add_scalar(
            "Accuracy/Train", self._train_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Accuracy/Val", self._val_acc[-1], self._current_epoch_num
        )
        self._writer.add_scalar(
            "Loss/Train", self._train_loss[-1], self._current_epoch_num
        )
        self._writer.add_scalar("Loss/Val", self._val_loss[-1], self._current_epoch_num)

        print(message)

    def _calc_acc_on_private_test(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test..")

        with torch.no_grad():
            for i, (images, targets) in tqdm(
                    enumerate(self._test_loader), total=len(self._test_loader), leave=False
            ):
                # TODO: implement augment when predict
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                outputs = self._model(images)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        return test_acc

    def _calc_acc_on_private_test_with_tta(self):
        self._model.eval()
        test_acc = 0.0
        print("Calc acc on private test with tta..")

        # for idx in len(self._test_set):
        #     image, label = self._test_set[idx]

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self._test_loader), total=len(self._test_loader), leave=False):
                # TODO: implement augment when predict
                # images = images.to(self._device)

                # targets = targets.to(self._device) # .to(non_blocking=True)

                images = batch["image"].to(self._device)  # .cuda(non_blocking=True)
                targets = batch["label"].to(self._device)

                outputs = self._model(images)
                acc = accuracy(outputs, targets)[0]
                test_acc += acc.item()

            test_acc = test_acc / (i + 1)
        print("Accuracy on private test: {:.3f}".format(test_acc))
        return test_acc

    def train(self):
        print("start training")
        # print(self._model)
        while not self._is_stop():
            self._increase_epoch_num()
            self._train()
            self._val()

            self._update_training_state()
            self._logging()

        # training stop and then load the checkpoint
        # and produce masks
        try:
            state = torch.load(self._checkpoint_path)
            if self._distributed:
                self._model.module.load_state_dict(state["net"])
            else:
                self._model.load_state_dict(state["net"])
            test_acc = self._calc_acc_on_private_test_with_tta()
            self._save_weights()
        except Exception as e:
            print("Testing error when training stop")
            print(e)

        self._writer.add_text(
            "Summary", "Converged after {} epochs".format(self._current_epoch_num)
        )
        self._writer.add_text(
            "Summary",
            "Best validation accuracy: {:.3f}".format(self._current_epoch_num),
        )
        self._writer.add_text(
            "Summary", "Private test accuracy: {:.3f}".format(self._test_acc)
        )
        self._writer.close()