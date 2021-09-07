import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import datetime
import os


def train(training_method, model, device, optimizer, loss_function, n_epochs, train_set, test_set, metrics):
    if training_method == "supervised":
        return train_supervised(model, device, optimizer, loss_function, n_epochs, train_set, test_set, metrics)
    elif training_method == "swav":
        return train_swav(model, train_set)
    elif training_method == "simclr":
        return train_simclr(model, train_set)
    else:
        raise NotImplementedError()


def train_supervised(model, device, optimizer, loss_function, n_epochs, train_set, test_set, metrics):
    print("Started training")
    save_log = os.path.join('/scratch/users/rvandeghen/trail/pretrained_sup_imagenet_ds_xray', 'tensorboard', datetime.datetime.now().strftime("%m%d-%H%M%S"))
    print('Saving tensorboard at {}'.format(save_log))
    writer = SummaryWriter(save_log)
    history = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        evaluation = []
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
            if True: #i % 10 == 9:
                running_loss = running_loss / 10
                accuracy = np.mean(evaluation)
                log_step = epoch * len(train_set) + i
                history.append((log_step, running_loss, accuracy))
                print('[%d, %5d/%d] loss: %.3f\t accuracy: %.3f' %
                      (epoch + 1, i + 1, len(train_set), running_loss, accuracy))
                writer.add_scalar("Running Loss/train", running_loss, log_step)
                writer.add_scalar("Accuracy/train", accuracy, log_step)
                evaluation = []
                running_loss = 0.0
        val_loss, val_accuracy = _validate(test_set, model, device, loss_function)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    # writer.flush()
    # writer.close()
    print('Finished Training')
    return history


def train_swav(model, dataset):
    raise NotImplementedError()


def train_simclr(model, dataset):
    raise NotImplementedError()


def _validate(val_dataset, model, device, loss_function):
    validation_loss = 0.0
    num_correct = 0
    num_items = 0
    for _, data in enumerate(val_dataset):
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
