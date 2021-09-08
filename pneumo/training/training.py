import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger
import datetime
import os
import torch
import pytorch_lightning as pl


def train(model, train_set, val_set):
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_set, val_set)


def train_supervised(model, device, optimizer, loss_function, n_epochs, train_set, val_set, metrics):
    print("Started training")
    save_log = os.path.join('/scratch/users/rvandeghen/trail/pretrained_sup_imagenet_ds_xray',
                            'tensorboard', datetime.datetime.now().strftime("%m%d-%H%M%S"))
    print('Saving tensorboard at {}'.format(save_log))
    writer = SummaryWriter(save_log)
    history = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        evaluation = []
        model.train()
        for i, data in enumerate(train_set):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            evaluation.append((outputs.argmax(1) == labels).cpu().sum().item() / len(outputs))
            running_loss += loss.item()
            if True:  # i % 10 == 9:
                running_loss = running_loss / 10
                accuracy = np.mean(evaluation)
                log_step = epoch * len(train_set) + i
                history.append((log_step, running_loss, accuracy))
                print('[%d, %5d/%d] loss: %.3f\t accuracy: %.3f' %
                      (epoch + 1, i + 1, len(train_set), running_loss, accuracy))
                writer.add_scalars("Loss", {'Train': running_loss}, log_step)
                writer.add_scalars("Accuracy", {'Train': accuracy}, log_step)
                evaluation = []
                running_loss = 0.0
        val_loss, val_accuracy = _validate(val_set, model, device, loss_function)
        writer.add_scalars("Loss", {'Eval': val_loss}, epoch)
        writer.add_scalars("Accuracy", {'Eval': val_accuracy}, epoch)
    # writer.flush()
    # writer.close()
    print('Finished Training')
    return history


def _validate(val_dataset, model, device, loss_function):
    validation_loss = 0.0
    num_correct = 0
    num_items = 0
    model.eval()
    for _, data in enumerate(val_dataset):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            validation_loss += loss.item()
            num_correct += (outputs.argmax(1) == labels).sum().cpu().item()
            num_items += len(outputs)

    validation_loss = validation_loss / len(val_dataset)
    accuracy = num_correct / num_items
    return validation_loss, accuracy
