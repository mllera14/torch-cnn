# coding: utf-8

import sys
import os
import argparse
import time
import datetime
import csv
import yaml

import numpy as np
import torch
import torch.nn as nn

from ignite.engine import Events
from ignite.metrics import Loss, Accuracy

import tools
from cnn import LeNet

########################################################################################
# PARSE THE INPUT
########################################################################################


parser = argparse.ArgumentParser(description='PyTorch MNIST CNN model test')

# Model parameters
parser.add_argument('--model', type=str, default='LeNet',
    help='CNN model tu use. One of LeNet|AlexNet')

# Data parameters
parser.add_argument('--data', type=str, default='',
    help='location of the data set')
parser.add_argument('--train-test-split', type=float, default=0.2,
    help='proportion of trainig data used for validation')
parser.add_argument('--shuffle', action='store_true',
    help='shuffle the data at the start of each epoch.')
parser.add_argument('--input-size', type=int, default=1,
    help='the default dimensionality of each input timestep.'
    'defaults to 1, meaning instances are treated like one large 1D sequence')

# Training parameters
parser.add_argument('--epochs', type=int, default=40,
    help='max number of training epochs')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
    help='batch size')
parser.add_argument('--optim', type=str, default='rmsprop',
    help='learning rule, supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--lr', type=float, default=1e-4,
    help='initial learning rate')
parser.add_argument('--l2-norm', type=float, default=0,
    help='weight of L2 norm')
parser.add_argument('--clip', type=float, default=1,
    help='gradient clipping')
parser.add_argument('--track-hidden', action='store_true',
    help='keep the hidden state values across a whole epoch of training')

# Replicability and storage
parser.add_argument('--save', type=str,  default='results',
    help='path to save the final model')
parser.add_argument('--seed', type=int, default=18092,
    help='random seed')

# CUDA
parser.add_argument('--cuda', action='store_true',
    help='use CUDA')

# Print options
parser.add_argument('--verbose', action='store_true',
    help='print the progress of training to std output.')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
    help='report interval')

args = parser.parse_args()

########################################################################################
# SETTING UP THE DIVICE AND SEED
########################################################################################

torch.manual_seed(args.seed)
if args.cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

########################################################################################
# LOAD DATA
########################################################################################

# # Load data
data_path, batch_size = args.data, args.batch_size

train_data, test_data, validation_data = tools.load_mnist(
    data_path=data_path,
    input_size=args.input_size,
    batch_size=batch_size,
    val_split_ratio=0.2,
    shuffle=True,
    download=False
)

########################################################################################
# CREATE THE MODEL
########################################################################################

if args.model == 'LeNet':
    model = LeNet()
else:
    raise ValueError('Unrecognized model')

input_size, n_classes = args.input_size, 10
model_params = {'model_type': args.model}


########################################################################################
# SET UP OPTIMIZER & OBJECTIVE FUNCTION
########################################################################################

optimizer = tools.init_optimizer(args.optim, model.parameters(), args.lr, args.l2_norm)
criterion = nn.CrossEntropyLoss()

########################################################################################
# TRAINING SETUP
########################################################################################

epochs, log_interval, save_path = args.epochs, args.log_interval, args.save
metrics = {'xent': Loss(criterion), 'accuracy': Accuracy()}

trainer = tools.create_feedforward_trainer(
    model, optimizer, criterion, grad_clip=args.clip, device=device)
validator = tools.create_feedforward_evaluator(model, metrics, device=device)

@trainer.on(Events.EPOCH_COMPLETED)
def validate(engine):
    validator.run(validation_data)

# Add handlers. Learning rate decay
lr_scheduler = tools.LRSchedulerHandler(
    tools.init_lr_scheduler(
        optimizer, 'reduce-on-plateau',
        lr_decay=0.1, patience=10),
    'xent').attach(validator)

# Tracing
tracer = tools.TracerHandler(metrics.keys()).attach(trainer, validator)
if args.verbose:
    logger = tools.LoggerHandler(train_data, args.log_interval
        ).attach(trainer, validator, metrics.keys())

# Early stopping and model checkpoint
def score_fn(engine):
    return -engine.state.metrics['xent']

stopper = tools.EarlyStopping(
    patience=50,
    score_function=score_fn,
    trainer=trainer
)

checkpoint = tools.ModelCheckpoint(
    dirname=save_path,
    filename_prefix='',
    score_function=score_fn,
    create_dir=True,
    require_empty=False,
    save_as_state_dict=True
)

validator.add_event_handler(Events.COMPLETED, stopper)
validator.add_event_handler(Events.COMPLETED, checkpoint, {'model': model})

# Training time
timer = tools.Timer(average=False)
timer.attach(trainer)

########################################################################################
# RUN TRAINING
########################################################################################

# Train model

trainer.run(train_data, max_epochs=epochs)

# Test
best_model_path = str(checkpoint._saved[-1][1][0])
with open(best_model_path, mode='rb') as f:
    state_dict = torch.load(f)
model.load_state_dict(state_dict)

tester = tools.create_feedforward_evaluator(model, metrics, device=device)
tester.run(test_data)

# Testing preformance
test_loss = tester.state.metrics['xent']
test_acc = tester.state.metrics['accuracy']

print('Training ended: test loss {:5.4f} - test accuracy {:3.2%}'.format(
    test_loss, test_acc))

########################################################################################
# Save results
########################################################################################

print('Saving results....')

learning_params = {
    'optimizer': args.optim,
    'learning-rate': args.lr,
    'l2-norm': args.l2_norm,
    'criterion': 'xent'
}

meta = {
    'learning-params': learning_params,
    'info': {
        'test-score': test_loss,
        'accuracy': test_acc,
        'training-time': timer.value(),
        'timestamp': datetime.datetime.now()
    }
}

with open(save_path + '/meta.yaml', mode='w') as f:
    yaml.dump(meta, f)

# Save traces
tracer.save_traces(save_path)

print('Done.')
