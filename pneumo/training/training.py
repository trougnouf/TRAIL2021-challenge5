import os
from torch.utils.tensorboard import SummaryWriter
import torch
import pytorch_lightning as pl
from tqdm import tqdm


def train_ssl(model, train_set, val_set, device, nepochs):
    if device == "cuda":
        gpus = 1
    else:
        gpus = None
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=nepochs,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_set, val_set)


def train_supervised(model, device, optimizer, loss_function, n_epochs, train_set, val_set, experiment_name):
    print("Started training")
    device = torch.device(device)
    model.to(device)
    writer = SummaryWriter()
    best_loss = float("inf")
    history = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"Epoch #{epoch}")
        running_loss = 0.0
        model.train()
        num_correct = 0
        num_items = 0
        pbar = tqdm(enumerate(train_set), total=len(train_set))
        for i, data in pbar:
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
            num_correct_batch = (outputs.argmax(1) == labels).sum().item()
            num_correct += num_correct_batch
            num_items += len(outputs)
            running_loss += loss.item()
            pbar.set_description('loss: %.3f - accuracy: %.3f' % (loss.item(), num_correct_batch / len(outputs)))

            if i % 10 == 9:
                running_loss = running_loss / 10
                accuracy = num_correct / num_items
                log_step = epoch * len(train_set) + i
                history.append((log_step, running_loss, accuracy))
                writer.add_scalar("Loss/train", running_loss, log_step)
                writer.add_scalar("Accuracy/train", accuracy, log_step)
                num_items = 0
                num_correct = 0
                running_loss = 0.0
        val_loss, val_accuracy = _validate(val_set, model, device, loss_function)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        if val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(writer.get_logdir(), experiment_name + ".chkpt"))
    writer.flush()
    writer.close()
    print('Finished Training')
    return history


def _validate(val_dataset, model, device, loss_function):
    validation_loss = 0.0
    num_correct = 0
    num_items = 0
    model.eval()
    print("Validation")
    pbar = tqdm(val_dataset)
    for data in pbar:
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
    pbar.set_description('loss: %.3f - accuracy: %.3f' % (validation_loss, accuracy))
    return validation_loss, accuracy
