import argparse

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchfunc
import numpy as np
import random
import os
import shutil

from models.resnet import resnet50
from data.data_manager import get_val_loader, get_train_loader

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")

    parser.add_argument("--img_dir", default='D:/Work/GAN/FFHQ/images', help="Images dir path")
    parser.add_argument("--train_json_path", default='D:/Work/GAN/FFHQ/train.json', help="Train json path")
    parser.add_argument("--val_json_path", default='D:/Work/GAN/FFHQ/val.json', help="Validation json path")
    parser.add_argument("--test_json_path", default='D:/Work/GAN/FFHQ/test.json', help="Test json path")

    parser.add_argument("--checkpoint", default='checkpoints', help="Logs dir path")
    parser.add_argument("--log_dir", default='logs', help="Logs dir path")
    parser.add_argument("--log_prefix", default='', help="Logs dir path")

    return parser.parse_args()

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = resnet50(pretrained=args.pretrained, num_classes=1)
        self.model = model.to(device)

        self.train_loader = get_train_loader(args)
        self.val_loader = get_val_loader(args)

        self.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(args.epochs * .3))

        self.criterion = nn.BCEWithLogitsLoss()

        self.writer = SummaryWriter(log_dir=str(args.log_dir))
        self.best_acc = 0

    def _do_epoch(self, epoch_idx):
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            # outputs = torch.squeeze(outputs)

            loss = self.criterion(outputs, targets)

            if batch_idx % 30 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, '
                      f'loss: {loss.item()}')

            self.writer.add_scalar('loss_train', loss.item(), epoch_idx * len(self.train_loader) + batch_idx)

            loss.backward()
            self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            total = len(self.val_loader.dataset)
            class_correct = self.do_test(self.val_loader)
            class_acc = float(class_correct) / total
            print(f'Validation Accuracy: {class_acc}')

            is_best = False
            if class_acc > self.best_acc:
                self.best_acc = class_acc
                is_best = True

            checkpoint_name = f'checkpoint_{epoch_idx + 1}_acc_{round(class_acc, 3)}.pth.tar'
            print(f'Saving {checkpoint_name} to dir {self.args.checkpoint}')
            save_checkpoint({
                'epoch': epoch_idx + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, checkpoint=self.args.checkpoint, filename=checkpoint_name)

            self.writer.add_scalar('val_accuracy', class_acc, epoch_idx)

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward
            outputs = self.model(inputs)

            cls_pred = (outputs > 0.5).type(torch.int)
            labels = (targets > 0.5).type(torch.int)

            class_correct += torch.sum(cls_pred == labels)

        return class_correct

    def do_training(self):
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self._do_epoch(self.current_epoch)

        self.writer.close()

        return self.best_acc

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(args, device)
    best_val_acc = trainer.do_training()

if __name__ == "__main__":
    torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()