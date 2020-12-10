import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from pathlib import Path
from logging import getLogger


class Solver:

    def __init__(self, net, train_dataloader, val_dataloader, test_dataloader, optimizer,
            scheduler, epochs, device, save_dir, lossfunc):

        self.net = net.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.save_dir = Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir)
        self.lossfunc = lossfunc

        torch.backends.cudnn.benchmark = True

        self.writer = SummaryWriter(log_dir=".")
        self.logger = getLogger("Solver")

        summary(self.net, (1,28,28))

    def save(self):
        torch.save(self.net.state_dict(), str(self.save_dir / "model.pt"))

    def test (self):
        test_loss = 0
        test_acc = 0
        num_count = 0
        with torch.no_grad():
            for idx, (image, ans) in enumerate(self.test_dataloader):
                # ans = nn.functional.one_hot(ans, num_classes=10).type(torch.LongTensor)
                image, ans = image.to(self.device), ans.to(self.device)
                out = self.best_net(image)
                loss = self.lossfunc(out, ans)
                test_loss += loss.sum().item()
                test_acc += self._get_acc(out, ans)
                num_count += image.size()[0]

        test_loss /= num_count
        test_acc /= num_count
        return {"loss": test_loss, "acc": test_acc }

    def _get_acc(self, out, ans):
        # ans = nn.functional.one_hot(ans, num_classes=10).type(torch.LongTensor)
        out = out.argmax(dim=1, keepdim=True)
        correct = out.eq(ans.view_as(out)).sum().item()
        return correct

    def train(self):
        best_loss = 1000
        for epoch in range(self.epochs):
            res = self._train_epoch()
            val_loss = res["val_loss"]
            if best_loss > val_loss:
                self.best_net = self.net
            self.logger.info("epoch: {} train loss :{:.4f}".format(epoch, res["train_loss"]))
            self.logger.info("epoch: {} train acc: {:.4f}".format(epoch, res["train_acc"]))
            self.logger.info("epoch: {} val loss: {:.4f}".format(epoch, res["val_loss"]))
            self.logger.info("epoch: {} val acc: {:.4f}".format(epoch, res["val_acc"]))
            self.scheduler.step()
            self.writer.add_scalar("train_loss", res["train_loss"], epoch)
            self.writer.add_scalar("train_acc", res["train_acc"], epoch)
            self.writer.add_scalar("val_loss", res["val_loss"], epoch)
            self.writer.add_scalar("val_acc", res["train_acc"], epoch)
        return {"train_loss": res["train_loss"], "train_acc": res["train_acc"],
                "val_loss": res["val_loss"], "val_acc":res["val_acc"]}

    def _train_epoch(self):
        train_correct=0
        train_loss = 0
        num_data = 0
        for idx, (image, ans) in enumerate(self.train_dataloader):
            # ans = nn.functional.one_hot(ans, num_classes=10).type(torch.LongTensor)
            # print(ans)
            self.optimizer.zero_grad()
            image, ans = image.to(self.device), ans.to(self.device)
            out = self.net(image)
            loss = self.lossfunc(out, ans)
            loss.backward()
            self.optimizer.step()
            train_correct += self._get_acc(out, ans)
            train_loss += loss.sum().item()
            num_data += image.size()[0]

        train_loss = train_loss / num_data
        train_acc = train_correct / num_data

        with torch.no_grad():
            val_loss =0
            val_correct = 0
            num_data = 0
            for idx, (image, ans) in enumerate(self.val_dataloader):
                # ans = nn.functional.one_hot(ans, num_classes=10).type(torch.LongTensor)
                image, ans = image.to(self.device), ans.to(self.device)
                out = self.net(image)
                loss = self.lossfunc(out, ans)
                val_loss += loss.sum().item()
                val_correct += self._get_acc(out, ans)
                num_data += image.size()[0]

        val_loss = val_loss / num_data
        val_acc = val_correct / num_data
        return {"train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc}

