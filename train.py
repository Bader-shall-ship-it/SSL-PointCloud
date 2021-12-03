import torch
from losses import frankenstein_contrastive_loss, contrastive_loss
import utils
from trainer import ContrastiveTrainer, SupervisedTrainer

def train_simclr(trainer: ContrastiveTrainer, data, left, right, epochs=2, batch_size=64, scheduler=None, save_path="./", save_name='simCLR_model_') -> None:
    for epoch in range(epochs):
        trainer.train_epoch(data, left, right, batch_size)

        if scheduler != None:
            scheduler.step()
        utils.save_model(trainer.model, epoch, 0, save_path, save_name)

def train_supervised(trainer: SupervisedTrainer, data, labels: torch.Tensor, epochs: int=2, batch_size=64, scheduler=None, save_path="./", save_name='simCLR_model_') -> None:
    best_acc = 0
    accuracies = []
    for epoch in range(epochs):
        # print("[{}/{}] Starting new epoch".format(epoch, epochs))
        trainer.train_epoch(data, labels, batch_size)

        if scheduler != None:
            scheduler.step()

        curr_loss, curr_acc = trainer.evaluate(data, labels)
        print('[%d/%d] loss: %f accuracy: %f' % (epoch, epochs, curr_loss.item(), curr_acc.item()))
        if curr_acc > best_acc:
            best_acc = curr_acc
            utils.save_model(trainer.model, epoch, 0, save_path, "best_acc"+ save_name)
        utils.save_model(trainer.model, epoch, 0, save_path, save_name)

        accuracies += [curr_acc.item()]
