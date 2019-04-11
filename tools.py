import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as trans
import torchvision.datasets as dataset
from ignite.engine import Events, Engine, _prepare_batch
from ignite.handlers import *

########################################################################################
# Training
########################################################################################

def create_feedforward_trainer(model, optimizer, loss_fn, grad_clip=0, device=None,
                    non_blocking=False, prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _training_loop(engine, batch):
        # Set model to training and zero the gradients
        model.train()
        optimizer.zero_grad()

        # Load the batches
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)

        # Forward pass
        pred = model(inputs)
        loss = loss_fn(pred, targets)

        # Backwards
        loss.backward()

        # Optimize
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        return loss.item()

    return Engine(_training_loop)


########################################################################################
# Testing
########################################################################################

def create_feedforward_evaluator(model, metrics, device=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(
                batch, device=device, non_blocking=non_blocking)
            output = model(inputs)
            return output, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


########################################################################################
# Optimizer
########################################################################################

def init_optimizer(optimizer, params, lr=0.01, l2_norm=0.0, **kwargs):

    if optimizer == 'adam':
        optimizer = optim.Adam(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'adamax':
        optimizer = optim.Adamax(params,
            lr=lr, eps=1e-9, weight_decay=l2_norm, betas=[0.9, 0.98])
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params,
            lr=lr, eps=1e-10, weight_decay=l2_norm, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params,
            lr=lr, weight_decay=l2_norm, momentum=0.9) # 0.01
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params,
            lr=lr, weight_decay=l2_norm, lr_decay=0.9)
    elif optimizer == 'adadelta':
        optimizer = optim.Adadelta(params,
            lr=lr, weight_decay=l2_norm, rho=0.9)
    else:
        raise ValueError(r'Optimizer {0} not recognized'.format(optimizer))

    return optimizer


def init_lr_scheduler(optimizer, scheduler, lr_decay, patience,
        threshold=1e-4, min_lr=1e-9):

    if scheduler == 'reduce-on-plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=lr_decay,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
    else:
        raise ValueError(r'Scheduler {0} not recognized'.format(scheduler))

    return scheduler

########################################################################################
# Handlers
########################################################################################

class LRSchedulerHandler(object):
    def __init__(self, scheduler, loss):
        self.scheduler = scheduler
        self.loss = loss

    def __call__(self, engine):
        loss_val = engine.state.metrics[self.loss]
        self.scheduler.step(loss_val)

    def attach(self, engine):
        engine.add_event_handler(Events.COMPLETED, self)
        return self


class TracerHandler(object):
    def __init__(self, val_metrics):
        self.metrics = ['loss']
        self.loss = []
        self._batch_trace = []

        template = 'val_{}'
        for k in val_metrics:
            name = template.format(k)
            setattr(self, name, [])
            self.metrics.append(name)

    def _initalize_traces(self, engine):
        for k in self.metrics:
            getattr(self, k).clear()

    def _save_batch_loss(self, engine):
        self._batch_trace.append(engine.state.output)

    def _trace_training_loss(self, engine):
        avg_loss = np.mean(self._batch_trace)
        self.loss.append(avg_loss)
        self._batch_trace.clear()

    def _trace_validation(self, engine):
        metrics = engine.state.metrics
        template = 'val_{}'
        for k, v in metrics.items():
            trace = getattr(self, template.format(k))
            trace.append(v)

    def attach(self, trainer, evaluator=None):
        trainer.add_event_handler(Events.STARTED, self._initalize_traces)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._save_batch_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._trace_training_loss)

        if evaluator is not None:
            evaluator.add_event_handler(Events.COMPLETED, self._trace_validation)

        return self

    def save_traces(self, save_path):
        for loss in self.metrics:
            trace = getattr(self, loss)
            with open('{}/{}.csv'.format(save_path, loss), mode='w') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                for i, v in enumerate(trace):
                    wr.writerow([i + 1, v])


class LoggerHandler(object):
    def __init__(self, loader, log_interval, pbar=None, desc=None):
        n_batches = len(loader)
        self.desc = 'iteration-loss: {:.2f}' if desc is None else desc
        self.pbar = pbar or tqdm(
            initial=0, leave=False, total=n_batches,
            desc=self.desc.format(0)
        )
        self.log_interval = log_interval
        self.running_loss = 0
        self.n_batches = n_batches

    def _log_batch(self, engine):
        self.running_loss += engine.state.output

        iter = (engine.state.iteration - 1) % self.n_batches + 1
        if iter % self.log_interval == 0:
            self.pbar.desc = self.desc.format(
                engine.state.output)
            self.pbar.update(self.log_interval)

    def _log_epoch(self, engine):
        self.pbar.refresh()
        tqdm.write("Epoch: {} - avg loss: {:.2f}"
            .format(engine.state.epoch, self.running_loss / self.n_batches))
        self.running_loss = 0
        self.pbar.n = self.pbar.last_print_n = 0

    def _log_validation(self, engine):
        metrics = self.evaluator.state.metrics

        message = []
        for k, v in metrics.items():
            message.append("{}: {:.2f}".format(k, v))
        tqdm.write('\tvalidation: ' + ' - '.join(message))

    def attach(self, trainer, evaluator=None, metrics=None):
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_epoch)
        trainer.add_event_handler(Events.COMPLETED, lambda x: self.pbar.close())

        if evaluator is not None and metrics is None:
            raise ValueError('')

        if evaluator is not None:
            self.evaluator = evaluator
            trainer.add_event_handler(Events.EPOCH_COMPLETED, self._log_validation)

        return self

        for name, metric in metrics.items():
            metric.attach(engine, name)

        return engine

########################################################################################
# Data
########################################################################################


def load_raw(data_path, input_size, download=False):
    transform = trans.Compose([
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, input_size))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, download=download)

    return train_data, test_data


def load_mnist(data_path, input_size, batch_size, val_split_ratio,
        shuffle=True, download=False):
    train_raw, test_raw = load_raw(data_path, input_size, download)

    # Split train data into training and validation sets
    N = len(train_raw)
    val_size = int(N * val_split_ratio)
    train_raw, validation_raw = random_split(
        train_raw, [N - val_size, val_size])

    train_data = DataLoader(train_raw, batch_size=batch_size, shuffle=shuffle)
    validation_data = DataLoader(validation_raw, batch_size=val_size, shuffle=False)
    test_data = DataLoader(test_raw, batch_size=len(test_raw), shuffle=False)

    return train_data, validation_data, test_data
